# ====== 調整用パラメータ ======
CAMERA_SIZE = (1920, 1080)      # 要求するカメラ解像度（非対応なら近い値になる）
MAX_CAMERA_PROBE = 10           # カメラ名が取れないときに探す番号の数
MAX_NUM_HANDS = 2               # 同時につかえる手の数
MIN_DETECTION_CONFIDENCE = 0.7  # 信頼度の下限値 (手の検知)
MIN_TRACKING_CONFIDENCE = 0.6   # 信頼度の下限値 (トラッキング)

# 描画系（ピクセル値は 720p 基準。実際の解像度に合わせて拡大縮小する）
DRAW_THICKNESS = 10
MIN_POINT_DIST_PX = 3           # これ未満の移動では線を伸ばさない（過密な点を除去）
MAX_STROKE_JUMP_PX = 150        # これを超える移動はストロークを分断

# 指先の平滑化（One Euro Filter）
FILTER_MIN_CUTOFF = 1.5         # 小さいほど静止時の手ぶれを抑える
FILTER_BETA = 0.01              # 大きいほど速い動きへの遅れを減らす

# ジェスチャ判定（ヒステリシスあり / 3D距離を手のひらの大きさに対する比で判定）
PINCH_LEVEL = 5                 # 操作検出感度の初期値（1～10, ↑↓キーで変更 / 大きいほど甘い）
PINCH_LEVEL_MAX = 10
PINCH_RATIO_BASE = 0.12         # 親指-人差し指の距離が BASE + STEP * 感度 未満でON
PINCH_RATIO_STEP = 0.03
PINCH_OFF_MARGIN = 0.16         # ON の比にこれを足した値を超えたらOFF
PINCH_ON_FRAMES = 3             # ON確定に必要な連続フレーム
PINCH_OFF_FRAMES = 3            # OFF確定に必要な連続フレーム

# UI（720p 基準）
SWATCH_SIZE = 64
SWATCH_GAP = 12
TEXT_SIZE = 16
SELECT_DWELL_SEC = 0.4          # 色の上にこの秒数とどまると選択
CLEAR_DWELL_SEC = 1.0           # 消去ボタンの上にこの秒数とどまると全消去

# トラッキング（指先基準）
MATCHING_DISTANCE = 0.15        # 前フレームの指先と同じ手とみなす最大距離（画面幅に対する割合）
TRACK_LIFETIME = 8              # 手が何フレーム見えなくなったら追跡をやめるか
# ============================

import contextlib
import functools
import math
import os
import platform
import signal
import sys
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageDraw, ImageFont

WINDOW_NAME = "WebCam Drawing"
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # BGR: 青 緑 赤 黄
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
# PyInstaller でビルドした exe では同梱ファイルの展開先から読む
APP_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(APP_DIR, "hand_landmarker.task")
IS_WINDOWS = platform.system() == "Windows"
# 日本語を表示できるフォント（最初に見つかったもの）
FONT_CANDIDATES = [
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
# waitKeyEx が返す矢印キーのコード（Windows / Linux / macOS）
KEYS_UP = (2490368, 65362, 63232)
KEYS_DOWN = (2621440, 65364, 63233)

# MediaPipe Hand Landmarker のランドマーク番号
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9

# OpenCV 自身のログは抑制（外部DLL由来の出力は別途サプレッサで対応）
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)


# ====== ユーティリティ ======
@contextlib.contextmanager
def suppress_native_output():
    """標準出力/標準エラーをOSレベルで一時的に捨てる。
    外部DLLが吐くI/W/Eログも可能な範囲で抑制する。
    """
    try:
        for stream in (sys.stdout, sys.stderr):
            if stream is not None:
                stream.flush()
        saved = [os.dup(1), os.dup(2)]
    except OSError:
        # GUI 版（コンソールなし）では捨てる先もないのでそのまま実行
        yield
        return
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in saved:
                os.close(fd)


class OneEuroFilter:
    """速度に応じて平滑化の強さを変えるフィルタ（遅いときは強く、速いときは弱く）"""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * np.linalg.norm(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


# ====== カメラ ======
def camera_names():
    """DirectShowの列挙順にカメラ名を返す（取得できなければ空リスト）"""
    if not IS_WINDOWS:
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        return []


def choose_camera(names):
    """カメラ名のボタンを並べたウィンドウで選ばせる。閉じられたら None"""
    if len(names) <= 1:
        return 0
    import tkinter as tk
    root = tk.Tk()
    root.title(WINDOW_NAME)
    root.resizable(False, False)
    chosen = []

    def pick(i):
        chosen.append(i)
        root.destroy()

    buttons = []
    for i, name in enumerate(names):
        b = tk.Button(root, text=name, width=36, pady=6, command=lambda i=i: pick(i))
        b.pack(fill="x", padx=12, pady=(12 if i == 0 else 0, 12 if i == len(names) - 1 else 6))
        buttons.append(b)
    # キーボードでも選べる：Tab/↑↓で移動、Enterで決定、Escで閉じる
    root.bind("<Return>", lambda e: root.focus_get().invoke())
    root.bind("<Down>", lambda e: root.focus_get().tk_focusNext().focus_set())
    root.bind("<Up>", lambda e: root.focus_get().tk_focusPrev().focus_set())
    root.bind("<Escape>", lambda e: root.destroy())
    root.eval("tk::PlaceWindow . center")
    root.after(0, root.focus_force)
    buttons[0].focus_set()
    root.mainloop()
    return chosen[0] if chosen else None


def show_error(message):
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(WINDOW_NAME, message)
    root.destroy()


class Camera:
    """別スレッドで読み続け、最新フレームだけを保持するキャプチャ"""

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    @classmethod
    def open(cls, index):
        with suppress_native_output():
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if IS_WINDOWS else cv2.VideoCapture(index)
            ok = cap.isOpened()
            if ok:
                # MJPG にすると高解像度でもフレームレートが落ちにくい
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, _ = cap.read()
        if not ok:
            cap.release()
            return None
        return cls(cap)

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.seq += 1

    def read(self):
        with self.lock:
            return self.frame, self.seq

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def open_camera(start, count, skip_start=False):
    """start から順に開けるカメラを探す。(番号, Camera) を返し、なければ (start, None)"""
    for k in range(1 if skip_start else 0, count):
        index = (start + k) % count
        cam = Camera.open(index)
        if cam is not None:
            return index, cam
    return start, None


# ====== 手の検出 ======
def create_landmarker():
    if not os.path.exists(MODEL_PATH):
        print("手検出モデルをダウンロードしています...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
        os.replace(MODEL_PATH + ".part", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        model = f.read()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    with suppress_native_output():
        return vision.HandLandmarker.create_from_options(options)


class HandTrack:
    def __init__(self):
        self.filter = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
        self.tip = None          # 生の指先位置（マッチング用）
        self.pos = None          # 平滑化した指先位置（描画用）
        self.missed = 0
        self.pinched = False
        self.on_count = 0
        self.off_count = 0
        self.last_pt = None      # 描画中ストロークの最終点
        self.hover = None        # とどまっているパレット項目
        self.hover_start = 0.0
        self.hover_done = False

    def update(self, det, t, pinch_level):
        tip, ratio = det
        self.tip = tip
        self.pos = self.filter(tip, t)
        self.missed = 0

        # ピンチ（つまみ）判定
        on_ratio = PINCH_RATIO_BASE + PINCH_RATIO_STEP * pinch_level
        if ratio < on_ratio:
            self.on_count += 1
            self.off_count = 0
            if self.on_count >= PINCH_ON_FRAMES:
                self.pinched = True
        elif ratio > on_ratio + PINCH_OFF_MARGIN:
            self.off_count += 1
            self.on_count = 0
            if self.off_count >= PINCH_OFF_FRAMES:
                self.pinched = False
        else:
            # ヒステリシス帯ではカウンタを減衰させる
            self.on_count = max(0, self.on_count - 1)
            self.off_count = max(0, self.off_count - 1)

    def point(self):
        return int(round(self.pos[0])), int(round(self.pos[1]))


def detect_hands(landmarker, frame_rgb, timestamp_ms):
    """各手の (指先のピクセル座標, ピンチ比) を返す。
    ピンチ比は親指-人差し指の3D距離 / 手のひらの大きさ。奥行きも見るので、
    カメラから見て指が重なっただけではつまんだと判定されない。
    """
    h, w = frame_rgb.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(image, timestamp_ms)
    dets = []
    for lms, world in zip(result.hand_landmarks, result.hand_world_landmarks):
        p3 = lambda i: np.array([world[i].x, world[i].y, world[i].z])
        palm = max(np.linalg.norm(p3(WRIST) - p3(MIDDLE_MCP)), 1e-6)
        ratio = np.linalg.norm(p3(THUMB_TIP) - p3(INDEX_TIP)) / palm
        dets.append((np.array([lms[INDEX_TIP].x * w, lms[INDEX_TIP].y * h]), ratio))
    return dets


def match_tracks(tracks, dets, frame_w, next_id):
    """指先どうしが近い順に、追跡中の手と検出結果を対応づける"""
    max_dist = MATCHING_DISTANCE * frame_w
    pairs = sorted(
        (np.linalg.norm(tr.tip - det[0]), tid, di)
        for tid, tr in tracks.items() for di, det in enumerate(dets)
    )
    assigned = {}
    for dist, tid, di in pairs:
        if dist > max_dist:
            break
        if tid not in assigned and di not in assigned.values():
            assigned[tid] = di
    for tid in list(tracks):
        if tid not in assigned:
            tracks[tid].missed += 1
            if tracks[tid].missed > TRACK_LIFETIME:
                del tracks[tid]
    for di in range(len(dets)):
        if di not in assigned.values():
            tracks[next_id] = HandTrack()
            assigned[next_id] = di
            next_id += 1
    return {tid: dets[di] for tid, di in assigned.items()}, next_id


# ====== 画面上のUI ======
CLEAR = "clear"
PALETTE_KEYS = [str(i + 1) for i in range(len(COLORS))] + ["Space"]


@functools.lru_cache(maxsize=None)
def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


@functools.lru_cache(maxsize=256)
def text_sprite(text, size):
    """縁取り付きの白文字を (BGR, アルファ) の画像にする。
    毎フレーム描き直さないようにキャッシュする。
    """
    font = load_font(size)
    stroke = max(1, size // 8)
    ascent, descent = font.getmetrics()
    width = math.ceil(font.getlength(text)) + 2 * stroke
    img = Image.new("RGBA", (width, ascent + descent + 2 * stroke))
    ImageDraw.Draw(img).text((stroke, stroke), text, font=font, fill=(255, 255, 255, 255),
                             stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    rgba = np.asarray(img)
    return np.ascontiguousarray(rgba[..., 2::-1]), rgba[..., 3:].astype(np.float32) / 255


def draw_text(frame, text, size, x, y, align="left"):
    """(x, y) を左上（align="right" なら右上）として文字を重ね、文字幅を返す"""
    bgr, alpha = text_sprite(text, size)
    th, tw = alpha.shape[:2]
    if align == "right":
        x -= tw
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + tw, fw), min(y + th, fh)
    if x0 < x1 and y0 < y1:
        roi = frame[y0:y1, x0:x1]
        a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
        roi[:] = (bgr[y0 - y:y1 - y, x0 - x:x1 - x] * a + roi * (1 - a)).astype(np.uint8)
    return tw


class Overlay:
    """画面上のUI。寸法は 720p 基準で、フレームの高さに合わせて拡大縮小する"""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.s = h / 720
        self.swatch = self.px(SWATCH_SIZE)
        self.gap = self.px(SWATCH_GAP)
        self.text_size = self.px(TEXT_SIZE)
        self.items = [
            (self.gap + i * (self.swatch + self.gap), self.gap, item)
            for i, item in enumerate(COLORS + [CLEAR])
        ]
        label_h = text_sprite("1", self.text_size)[1].shape[0]
        self.bottom = self.gap * 2 + self.swatch + label_h
        self.right = self.items[-1][0] + self.swatch + self.gap

    def px(self, v):
        """720p 基準のピクセル値を実際の解像度に合わせる"""
        return max(1, round(v * self.s))

    def hit(self, pt):
        x, y = pt
        for rx, ry, item in self.items:
            if rx <= x < rx + self.swatch and ry <= y < ry + self.swatch:
                return item
        return None

    def in_area(self, pt):
        return pt[0] < self.right and pt[1] < self.bottom

    def draw_palette(self, frame, current_color):
        n = self.swatch
        for (x, y, item), key in zip(self.items, PALETTE_KEYS):
            p1, p2 = (x, y), (x + n, y + n)
            if item == CLEAR:
                cv2.rectangle(frame, p1, p2, (60, 60, 60), -1)
                m = n // 4
                cv2.line(frame, (x + m, y + m), (x + n - m, y + n - m), (255, 255, 255), self.px(4), cv2.LINE_AA)
                cv2.line(frame, (x + n - m, y + m), (x + m, y + n - m), (255, 255, 255), self.px(4), cv2.LINE_AA)
            else:
                cv2.rectangle(frame, p1, p2, item, -1)
                if item == current_color:
                    cv2.rectangle(frame, p1, p2, (255, 255, 255), self.px(4))
            # キー操作をアイコンの下に
            tw = text_sprite(key, self.text_size)[1].shape[1]
            draw_text(frame, key, self.text_size, x + (n - tw) // 2, y + n)

    def draw_status(self, frame, pinch_level, camera_label):
        """右下：操作検出感度のゲージとカメラ切り替え"""
        size = self.text_size
        line_h = text_sprite("高", size)[1].shape[0]
        right = self.w - self.gap
        y_cam = self.h - self.gap - line_h
        y_sens = y_cam - line_h

        draw_text(frame, f"T  {camera_label}", size, right, y_cam, align="right")

        x = right - draw_text(frame, "高い", size, right, y_sens, align="right") - self.px(6)
        cell_w, cell_h, cell_gap = self.px(7), self.px(12), self.px(3)
        cy = y_sens + line_h // 2
        for i in range(PINCH_LEVEL_MAX, 0, -1):
            x0 = x - cell_w
            p1, p2 = (x0, cy - cell_h // 2), (x, cy + cell_h // 2)
            cv2.rectangle(frame, p1, p2, (0, 0, 0), self.px(3))
            cv2.rectangle(frame, p1, p2, (255, 255, 255) if i <= pinch_level else (90, 90, 90), -1)
            x = x0 - cell_gap
        draw_text(frame, "↑↓  操作検出感度  低い", size, x - self.px(6), y_sens, align="right")

    def draw_cursor(self, frame, track, color, now):
        pt = track.point()
        if track.pinched:
            r = self.px(DRAW_THICKNESS) // 2
            cv2.circle(frame, pt, r + self.px(2), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
            return
        cv2.circle(frame, pt, self.px(15), (255, 255, 255), self.px(1), cv2.LINE_AA)
        cv2.circle(frame, pt, self.px(13), color, self.px(2), cv2.LINE_AA)
        if track.hover is not None and not track.hover_done:
            # とどまっている間、指先のまわりにリングが満ちていく
            dwell = CLEAR_DWELL_SEC if track.hover == CLEAR else SELECT_DWELL_SEC
            progress = min(1.0, (now - track.hover_start) / dwell)
            r = self.px(22)
            cv2.ellipse(frame, pt, (r, r), -90, 0, 360 * progress, (255, 255, 255), self.px(4), cv2.LINE_AA)


# ====== メイン ======
def screen_size():
    if not IS_WINDOWS:
        return None
    import ctypes
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def main():
    if IS_WINDOWS:
        # 高DPI 画面で Windows に引き伸ばされてぼやけないようにする
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass

    names = camera_names()
    cam_count = len(names) or MAX_CAMERA_PROBE
    start_index = choose_camera(names)
    if start_index is None:
        return
    cam_index, cam = open_camera(start_index, cam_count)
    if cam is None:
        show_error("利用可能なWebカメラが見つかりませんでした。")
        return

    landmarker = create_landmarker()

    running = True

    def _handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def camera_label():
        return names[cam_index] if names else f"Camera {cam_index}"

    def set_title():
        cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} - {camera_label()}")

    set_title()

    draw_color = COLORS[0]
    pinch_level = PINCH_LEVEL
    canvas = None     # 描いた線（BGR）
    ink = None        # 線のある画素のマスク
    ui = None
    tracks = {}
    next_id = 0
    last_seq = 0
    last_ts = 0
    shown = False

    while running:
        frame, seq = cam.read()
        if seq != last_seq:
            last_seq = seq
            now = time.monotonic()
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if canvas is None:
                canvas = np.zeros_like(frame)
                ink = np.zeros((h, w), np.uint8)
            elif canvas.shape[:2] != (h, w):
                # カメラ切替で解像度が変わったら線を引き伸ばして引き継ぐ
                canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_NEAREST)
                ink = cv2.resize(ink, (w, h), interpolation=cv2.INTER_NEAREST)
            if ui is None or (ui.w, ui.h) != (w, h):
                ui = Overlay(w, h)

            # VIDEO モードはタイムスタンプが単調増加である必要がある
            last_ts = max(last_ts + 1, int(now * 1000))
            dets = detect_hands(landmarker, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), last_ts)
            matched, next_id = match_tracks(tracks, dets, w, next_id)

            for tid, det in matched.items():
                tr = tracks[tid]
                tr.update(det, now, pinch_level)
                pt = tr.point()

                # 描画：ピンチ中かつパレット外なら、前の点から線を伸ばす
                if tr.pinched and not ui.in_area(pt):
                    if tr.last_pt is None:
                        tr.last_pt = pt
                    else:
                        d = math.dist(pt, tr.last_pt)
                        if d > ui.px(MAX_STROKE_JUMP_PX):
                            tr.last_pt = pt
                        elif d >= ui.px(MIN_POINT_DIST_PX):
                            thickness = ui.px(DRAW_THICKNESS)
                            cv2.line(canvas, tr.last_pt, pt, draw_color, thickness)
                            cv2.line(ink, tr.last_pt, pt, 255, thickness)
                            tr.last_pt = pt
                else:
                    tr.last_pt = None

                # パレット：ピンチしていない指先がとどまったら選択
                item = None if tr.pinched else ui.hit(pt)
                if item != tr.hover:
                    tr.hover, tr.hover_start, tr.hover_done = item, now, False
                elif item is not None and not tr.hover_done:
                    dwell = CLEAR_DWELL_SEC if item == CLEAR else SELECT_DWELL_SEC
                    if now - tr.hover_start >= dwell:
                        tr.hover_done = True
                        if item == CLEAR:
                            canvas[:] = 0
                            ink[:] = 0
                        else:
                            draw_color = item

            cv2.copyTo(canvas, ink, frame)
            ui.draw_palette(frame, draw_color)
            ui.draw_status(frame, pinch_level, camera_label())
            for tid in matched:
                ui.draw_cursor(frame, tracks[tid], draw_color, now)
            cv2.imshow(WINDOW_NAME, frame)
            if not shown:
                # 最初の表示で、画面に収まる大きさにウィンドウを合わせる
                screen = screen_size()
                if screen:
                    fit = min(1.0, 0.85 * screen[0] / w, 0.85 * screen[1] / h)
                    cv2.resizeWindow(WINDOW_NAME, int(w * fit), int(h * fit))
                shown = True

        key = cv2.waitKeyEx(1)  # 矢印キーを受け取るため拡張コードで読む
        if key == ord(" ") and canvas is not None:  # スペースキーで全消去
            canvas[:] = 0
            ink[:] = 0
        elif key in KEYS_UP:  # ↑↓キーで操作検出感度を変更
            pinch_level = min(PINCH_LEVEL_MAX, pinch_level + 1)
        elif key in KEYS_DOWN:
            pinch_level = max(1, pinch_level - 1)
        elif ord("1") <= key < ord("1") + len(COLORS):  # 数字キーで色変更
            draw_color = COLORS[key - ord("1")]
        elif key in (ord("t"), ord("T")):  # Tキーで次のカメラへ切り替え
            new_index, new_cam = open_camera(cam_index, cam_count, skip_start=True)
            if new_cam is not None:
                cam.release()
                cam_index, cam = new_index, new_cam
                tracks.clear()
                last_seq = 0
                set_title()

        # ウィンドウのバツボタンで終了
        if shown and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
