# ====== 調整用パラメータ ======
CAMERA_FRAME_WIDTH = 1920
CAMERA_FRAME_HEIGHT = 1080
MAX_NUM_HANDS = 2            # 同時につかえる手の数
MIN_DETECTION_CONFIDENCE = 0.7  # 信頼度の下限値 (手の検知)
MIN_TRACKING_CONFIDENCE = 0.6   # 信頼度の下限値 (トラッキング)

# 描画系
DRAW_COLOR = (255, 0, 0)
DRAW_THICKNESS = 10
DRAW_HISTORY_MAX = 20000        # 総描画点の上限
SMOOTHING_ALPHA = 0.35          # 0～1 (大きいほど最新値を強く反映)
MIN_POINT_DIST_PX = 4           # ピクセル単位で追加する最小距離（不要な密な点を除去）
INTERP_MAX_DIST_PX = 120        # これ以下の距離なら中間点で補間（px）
MAX_JUMP_DISTANCE = 0.30        # 正規化座標での「大ジャンプ」分断閾値

# カラーパレットUI
COLOR_RECT_WIDTH = 80
COLOR_RECT_HEIGHT = 60
COLOR_HOVER_FRAMES = 6          # パレット上でこのフレーム数滞留で色変更（デバウンス）

# ジェスチャ判定（ヒステリシスあり）
PINCH_ON_DIST = 0.045           # 親指-人差し指の距離（正規化）でON
PINCH_OFF_DIST = 0.065          # OFFに戻す距離（> ONより大きく）
PINCH_ON_FRAMES = 3             # ON確定に必要な連続フレーム
PINCH_OFF_FRAMES = 3            # OFF確定に必要な連続フレーム
FINGER_UP_MARGIN = 0.015        # tip が pip よりどれだけ上なら「伸びている」とするか

# --- トラッキング用パラメータ ---
MATCHING_DISTANCE_THRESHOLD = 0.1  # 手をマッチングする際の最大距離 (画面サイズに対する割合)
TRACK_LIFETIME = 15                # 手が何フレーム見えなくなったら追跡をやめるか
# ============================

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import platform
import time
import threading
import signal
import os
import sys
import contextlib

# OpenCV 自身のログは抑制（外部DLL由来の出力は別途サプレッサで対応）
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

# ====== ユーティリティ ======
def inside_rect(xy, rect_pos, rect_size):
    x, y = xy
    rx, ry = rect_pos
    rw, rh = rect_size
    return (rx < x < rx + rw) and (ry < y < ry + rh)

def is_finger_up(landmarks, tip_idx, pip_idx, margin=FINGER_UP_MARGIN):
    """tip が pip より十分上（yが小さい）なら True。
    MediaPipe の座標系は y が下向きに増えるので注意。
    """
    tip = landmarks.landmark[tip_idx]
    pip = landmarks.landmark[pip_idx]
    return (tip.y + margin) < pip.y

@contextlib.contextmanager
def suppress_console_output():
    """標準出力/標準エラーをOSレベルで一時的に/dev/nullへ。
    外部DLLが吐くI/W/Eログも可能な範囲で抑制する。
    """
    try:
        # Pythonのストリーム
        old_stdout, old_stderr = sys.stdout, sys.stderr
        devnull = open(os.devnull, 'w')
        # OSのファイルディスクリプタを退避
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            # 元に戻す
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            devnull.close()
    except Exception:
        # 何かあれば単に素通し
        yield

# ====== カメラ列挙（Windowsは名称取得対応） ======
def list_cameras(max_index=10):
    is_windows = platform.system() == 'Windows'
    backend = cv2.CAP_DSHOW if is_windows else 0
    available = []
    for i in range(max_index + 1):
        with suppress_console_output():
            cap = cv2.VideoCapture(i, backend) if is_windows else cv2.VideoCapture(i)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    available.append(i)
            cap.release()
    return available

def list_cameras_with_names(max_index=10):
    is_windows = platform.system() == 'Windows'
    if is_windows:
        try:
            # DirectShowの列挙順に名称を取得
            from pygrabber.dshow_graph import FilterGraph
            names = FilterGraph().get_input_devices()  # list[str]
            available = []
            for i, name in enumerate(names):
                with suppress_console_output():
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        ok, _ = cap.read()
                        if ok:
                            available.append((i, name))
                    cap.release()
            return available
        except Exception:
            pass  # フォールバックに移行
    # 非Windows or 失敗時は番号のみ
    idx = list_cameras(max_index)
    return [(i, f"Camera {i}") for i in idx]

def select_camera():
    cams = list_cameras_with_names(10)
    if not cams:
        print("利用可能なWebカメラが見つかりませんでした。")
        return None, None
    print("利用可能なカメラ:")
    for i, name in cams:
        print(f"  {i}: {name}")
    default_cam = cams[0][0]
    while True:
        try:
            s = input(f"使用するカメラ番号を入力してください [{default_cam}]: ").strip()
            selected = int(s) if s else default_cam
            valid_indices = [i for i, _ in cams]
            if selected in valid_indices:
                # 選択した名称を解決
                selected_name = dict(cams)[selected]
                return selected, selected_name
            print("リストにある番号を入力してください。")
        except ValueError:
            print("数字を入力してください。")

class VideoCaptureThread:
    def __init__(self, src=0, backend=None):
        self.src = src
        self.backend = backend
        with suppress_console_output():
            if backend is not None:
                self.cap = cv2.VideoCapture(src, backend)
            else:
                self.cap = cv2.VideoCapture(src)
        self.stopped = False
        self.lock = threading.Lock()
        self.frame = None
        # プリウォームして1フレーム確保
        with suppress_console_output():
            ret, f = self.cap.read()
        if ret:
            self.frame = f

    def start(self):
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, f = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass

def ar_drawing_game(camera_index, camera_name=""):
    mp_hands = mp.solutions.hands
    with suppress_console_output():
        hands = mp_hands.Hands(
            max_num_hands=MAX_NUM_HANDS,
            model_complexity=0,         # 軽量モデル
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
    draw_color = DRAW_COLOR

    # --- トラッキング管理用の変数 ---
    tracked_hands = {}
    next_track_id = 0
    frame_count = 0
    persistent_strokes = deque(maxlen=DRAW_HISTORY_MAX)

    is_windows = platform.system() == 'Windows'
    backend = cv2.CAP_DSHOW if is_windows else None
    # キャプチャをスレッド化して待ち時間削減
    vc_thread = VideoCaptureThread(camera_index, backend).start()

    window_title = f"AR Drawing Game - {camera_name}" if camera_name else "AR Drawing Game"

    # --- graceful shutdown handling ---
    running = True

    def _handle_signal(sig, frame):
        nonlocal running
        running = False

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        # signal may fail on some platforms/threads; ignore
        pass

    colors = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255)}
    color_rects = []
    for i, (name, bgr) in enumerate(colors.items()):
        color_rects.append({"name": name, "pos": (i * (COLOR_RECT_WIDTH + 10) + 20, 10), "size": (COLOR_RECT_WIDTH, COLOR_RECT_HEIGHT), "bgr": bgr})

    while True:
        frame = vc_thread.read()
        if frame is None:
            time.sleep(0.01)
            continue
        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame.flags.writeable = True

        # --- 現在のフレームで検出された手の情報をリスト化 ---
        current_detected_hands = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist_pos = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                current_detected_hands.append({'landmarks': hand_landmarks, 'pos': np.array([wrist_pos.x, wrist_pos.y]), 'matched': False})

        # --- トラッキングとマッチング処理 ---
        unmatched_detections = list(range(len(current_detected_hands)))

        for track_id, track_data in list(tracked_hands.items()):
            best_match_idx = -1
            min_dist = MATCHING_DISTANCE_THRESHOLD
            for i in unmatched_detections:
                dist = np.linalg.norm(track_data['pos'] - current_detected_hands[i]['pos'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = i
            if best_match_idx != -1:
                detection = current_detected_hands[best_match_idx]
                track_data['pos'] = detection['pos']
                track_data['landmarks'] = detection['landmarks']
                track_data['last_seen'] = frame_count
                detection['matched'] = True
                unmatched_detections.remove(best_match_idx)

        for i in unmatched_detections:
            detection = current_detected_hands[i]
            tracked_hands[next_track_id] = {
                'pos': detection['pos'],
                'landmarks': detection['landmarks'],
                'last_seen': frame_count,
                'deque': deque(maxlen=5000),
                'prev_drawing': False,
                'smoothed_index': detection['pos'].copy(),
                # ヒステリシス付きピンチ状態
                'pinched': False,
                'pinch_on_count': 0,
                'pinch_off_count': 0,
                # カラー選択のデバウンス
                'hover_target': None,
                'hover_count': 0
            }
            next_track_id += 1

        # --- 手ごとのジェスチャー認識と描画 ---
        active_track_ids = list(tracked_hands.keys())
        for track_id in active_track_ids:
            if frame_count - tracked_hands[track_id]['last_seen'] > TRACK_LIFETIME:
                # 手の追跡が切れたら、その手のdequeをグローバル履歴に移してから削除
                td = tracked_hands[track_id]
                # dequeは newest が先頭になっているので、古い順にして永続履歴へ追加
                points = list(td['deque'])[::-1]
                for p in points:
                    persistent_strokes.append(p)
                del tracked_hands[track_id]
                continue

            track_data = tracked_hands[track_id]
            hand_landmarks = track_data['landmarks']

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # --- 指先の平滑化 (EMA) と大ジャンプ検出 ---
            current_index_norm = np.array([index_tip.x, index_tip.y])
            if 'smoothed_index' not in track_data or track_data['smoothed_index'] is None:
                track_data['smoothed_index'] = current_index_norm.copy()

            dist_norm = np.linalg.norm(track_data['smoothed_index'] - current_index_norm)
            if dist_norm > MAX_JUMP_DISTANCE:
                # 大ジャンプ -> ストロークを分断して平滑位置をリセット
                if track_data.get('prev_drawing', False):
                    track_data['deque'].appendleft(None)
                track_data['smoothed_index'] = current_index_norm.copy()
            else:
                alpha = SMOOTHING_ALPHA
                track_data['smoothed_index'] = (1 - alpha) * track_data['smoothed_index'] + alpha * current_index_norm

            smoothed_index = track_data['smoothed_index']
            ix, iy = int(smoothed_index[0] * w), int(smoothed_index[1] * h)

            index_is_up = is_finger_up(hand_landmarks,
                                       mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                       mp_hands.HandLandmark.INDEX_FINGER_PIP)
            middle_is_up = is_finger_up(hand_landmarks,
                                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                        mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
            ring_is_up = is_finger_up(hand_landmarks,
                                      mp_hands.HandLandmark.RING_FINGER_TIP,
                                      mp_hands.HandLandmark.RING_FINGER_PIP)
            pinky_is_up = is_finger_up(hand_landmarks,
                                       mp_hands.HandLandmark.PINKY_TIP,
                                       mp_hands.HandLandmark.PINKY_PIP)

            drawing = False

            # --- カラー選択（人差し指のみを伸ばす） ---
            only_index_up = index_is_up and not (middle_is_up or ring_is_up or pinky_is_up)
            if only_index_up and not track_data.get('pinched', False):
                cv2.circle(frame, (ix, iy), 14, (255, 255, 0), 2)
                # どのパレットにいるか判定
                hover = None
                for r in color_rects:
                    if inside_rect((ix, iy), r["pos"], r["size"]):
                        hover = r["name"]
                        break
                if hover is not None:
                    if track_data['hover_target'] == hover:
                        track_data['hover_count'] += 1
                    else:
                        track_data['hover_target'] = hover
                        track_data['hover_count'] = 1
                    if track_data['hover_count'] >= COLOR_HOVER_FRAMES:
                        # 実際の色変更
                        for r in color_rects:
                            if r['name'] == hover:
                                draw_color = r['bgr']
                                break
                        track_data['hover_count'] = 0
                else:
                    track_data['hover_target'] = None
                    track_data['hover_count'] = 0
            else:
                track_data['hover_target'] = None
                track_data['hover_count'] = 0

            # --- ピンチ（つまみ）で描画ON：ヒステリシス＋連続フレーム判定 ---
            distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
            if distance < PINCH_ON_DIST:
                track_data['pinch_on_count'] += 1
                track_data['pinch_off_count'] = 0
                if not track_data['pinched'] and track_data['pinch_on_count'] >= PINCH_ON_FRAMES:
                    track_data['pinched'] = True
            elif distance > PINCH_OFF_DIST:
                track_data['pinch_off_count'] += 1
                track_data['pinch_on_count'] = 0
                if track_data['pinched'] and track_data['pinch_off_count'] >= PINCH_OFF_FRAMES:
                    track_data['pinched'] = False
            else:
                # ヒステリシス帯ではカウンタを小さく減衰させる
                track_data['pinch_on_count'] = max(0, track_data['pinch_on_count'] - 1)
                track_data['pinch_off_count'] = max(0, track_data['pinch_off_count'] - 1)

            if track_data['pinched']:
                # 最小ピクセル距離を満たすかを確認して追加（過密点を除去）
                last_point = None
                if track_data['deque']:
                    if track_data['deque'][0] is not None:
                        last_point = track_data['deque'][0][0]

                should_add = True
                if last_point is not None:
                    dx = ix - last_point[0]
                    dy = iy - last_point[1]
                    dist_px = (dx * dx + dy * dy) ** 0.5
                    if dist_px < MIN_POINT_DIST_PX:
                        should_add = False

                if should_add:
                    # ---------- ここから補間処理 ----------
                    if last_point is not None:
                        dx = ix - last_point[0]
                        dy = iy - last_point[1]
                        dist_px = (dx * dx + dy * dy) ** 0.5
                        if dist_px > INTERP_MAX_DIST_PX:
                            # ピクセル単位の大ジャンプ：ストロークを分断
                            track_data['deque'].appendleft(None)
                        else:
                            # 線形補間で中間点を埋める（間隔は MIN_POINT_DIST_PX）
                            steps = max(1, int(dist_px / MIN_POINT_DIST_PX))
                            # 中間点を古い順 -> 新しい順に作成してから appendleft する
                            inter_points = []
                            for s in range(1, steps):
                                t = s / steps
                                xi = int(last_point[0] + t * dx)
                                yi = int(last_point[1] + t * dy)
                                inter_points.append(((xi, yi), draw_color, DRAW_THICKNESS))
                            # appendleft するときは古い順から行い、最後に最新点を追加
                            for p in reversed(inter_points):
                                track_data['deque'].appendleft(p)
                    # 実点を追加（常に追加）
                    track_data['deque'].appendleft(((ix, iy), draw_color, DRAW_THICKNESS))
                    drawing = True
                    # ---------- 補間処理ここまで ----------

            if track_data['prev_drawing'] and not drawing:
                track_data['deque'].appendleft(None)
            track_data['prev_drawing'] = drawing

        # --- UIと軌跡の描画 ---
        for r in color_rects:
            cv2.rectangle(frame, r["pos"], (r["pos"][0] + r["size"][0], r["pos"][1] + r["size"][1]), r["bgr"], -1)
            if r["bgr"] == draw_color:
                cv2.rectangle(frame, r["pos"], (r["pos"][0] + r["size"][0], r["pos"][1] + r["size"][1]), (255, 255, 255), 4)

        # カメラ名を表示
        # if camera_name:
        #     cv2.putText(frame, f"Camera: {camera_name}", (20, COLOR_RECT_HEIGHT + 40),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

        for track_data in tracked_hands.values():
            dq = track_data['deque']
            for i in range(1, len(dq)):
                if dq[i - 1] is not None and dq[i] is not None:
                    p1, color1, thick1 = dq[i - 1]
                    p2, color2, thick2 = dq[i]
                    cv2.line(frame, p1, p2, color1, thick1)

        ps = list(persistent_strokes)
        for i in range(1, len(ps)):
            if ps[i - 1] is not None and ps[i] is not None:
                p1, color1, thick1 = ps[i - 1]
                p2, color2, thick2 = ps[i]
                cv2.line(frame, p1, p2, color1, thick1)

        # --- フレームを拡大して表示 ---
        scale_factor = 1.5  # ウィンドウの拡大倍率
        frame_resized = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        cv2.imshow(window_title, frame_resized)
        key = cv2.waitKey(5) & 0xFF
        # if key == ord('q'):
        #     break
        if key == 32:  # スペースキーで全消去
            for t_id in tracked_hands:
                tracked_hands[t_id]['deque'].clear()
            persistent_strokes.clear()
        # 数字キーで色変更（1:青 2:緑 3:赤 4:黄）
        if key in (ord('1'), ord('2'), ord('3'), ord('4')):
            key_to_name = {ord('1'): 'blue', ord('2'): 'green', ord('3'): 'red', ord('4'): 'yellow'}
            name = key_to_name.get(key)
            for r in color_rects:
                if r['name'] == name:
                    draw_color = r['bgr']
                    break
        # ウィンドウのバツボタンやシグナルでの終了要求を検出
        # getWindowProperty が 0 未満や 0 の場合はウィンドウが閉じられた
        try:
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            # 一部環境では例外が出ることがあるが、その場合は安全に終了
            break
        if not running:
            break

    # release resources
    try:
        vc_thread.release()
    except Exception:
        pass
    hands.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    selected_camera_index, camera_name = select_camera()
    if selected_camera_index is not None:
        ar_drawing_game(selected_camera_index, camera_name)