# ====== 調整用パラメータ ======
CAMERA_FRAME_WIDTH = 1920
CAMERA_FRAME_HEIGHT = 1080
MAX_NUM_HANDS = 10
MIN_DETECTION_CONFIDENCE = 0.8  # 少し下げて検出されやすく
MIN_TRACKING_CONFIDENCE = 0.5
DRAW_COLOR = (255, 0, 0)
DRAW_THICKNESS = 10
ERASER_THICKNESS = 50
COLOR_RECT_WIDTH = 80
COLOR_RECT_HEIGHT = 60
CLEAR_BUTTON_POS = (1150, 10)
CLEAR_BUTTON_SIZE = (100, 60)
CLEAR_BUTTON_TEXT = "Clear All"

# --- トラッキング用パラメータ ---
MATCHING_DISTANCE_THRESHOLD = 0.1  # 手をマッチングする際の最大距離 (画面サイズに対する割合)
TRACK_LIFETIME = 15                # 手が何フレーム見えなくなったら追跡をやめるか
# ============================

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import subprocess
import re
import time

# (list_cameras_ffmpeg と select_camera 関数は変更なし)
def list_cameras_ffmpeg():
    try:
        result = subprocess.run(
            ['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', '""'],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        output = result.stderr
        camera_list = []
        in_video_section = False
        for line in output.splitlines():
            if "AVFoundation video devices:" in line: in_video_section = True; continue
            if "AVFoundation audio devices:" in line: in_video_section = False
            if in_video_section:
                m = re.search(r'\[(\d+)\] (.+)', line)
                if m: camera_list.append((int(m.groups()[0]), m.groups()[1].strip()))
        return camera_list
    except Exception as e:
        print("カメラ一覧の取得に失敗しました:", e); return []

def select_camera():
    cameras = list_cameras_ffmpeg()
    if not cameras: print("利用可能なWebカメラが見つかりませんでした。"); return None
    print("利用可能なカメラ一覧:")
    for idx, name in cameras: print(f"{idx}: {name}")
    while True:
        try:
            default_cam = cameras[0][0] if cameras else 0
            selected_str = input(f"使用するカメラ番号を入力してください [{default_cam}]: ")
            selected = int(selected_str or default_cam)
            if any(idx == selected for idx, _ in cameras): return selected
            else: print("リストにある番号を入力してください。")
        except ValueError: print("数字を入力してください。")

def ar_drawing_game(camera_index):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    mp_drawing = mp.solutions.drawing_utils

    draw_color = DRAW_COLOR

    # --- トラッキング管理用の変数 ---
    tracked_hands = {}  # {track_id: data} 形式の辞書
    next_track_id = 0   # 次に割り当てる一意のID
    frame_count = 0     # フレームカウンター

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)

    colors = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255)}
    color_rects = []
    for i, (name, bgr) in enumerate(colors.items()):
        color_rects.append({"name": name, "pos": (i * (COLOR_RECT_WIDTH + 10) + 20, 10), "size": (COLOR_RECT_WIDTH, COLOR_RECT_HEIGHT), "bgr": bgr})
    clear_button = {"pos": CLEAR_BUTTON_POS, "size": CLEAR_BUTTON_SIZE, "text": CLEAR_BUTTON_TEXT}

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
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
                # 手首の座標を正規化された座標(0.0-1.0)で取得
                wrist_pos = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                current_detected_hands.append({'landmarks': hand_landmarks, 'pos': np.array([wrist_pos.x, wrist_pos.y]), 'matched': False})

        # --- トラッキングとマッチング処理 ---
        unmatched_detections = list(range(len(current_detected_hands)))
        
        # 1. 既存のトラックと今回の検出結果をマッチング
        for track_id, track_data in tracked_hands.items():
            best_match_idx = -1
            min_dist = MATCHING_DISTANCE_THRESHOLD
            
            for i in unmatched_detections:
                dist = np.linalg.norm(track_data['pos'] - current_detected_hands[i]['pos'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = i
            
            if best_match_idx != -1:
                # マッチ成功
                detection = current_detected_hands[best_match_idx]
                track_data['pos'] = detection['pos']
                track_data['landmarks'] = detection['landmarks']
                track_data['last_seen'] = frame_count
                detection['matched'] = True
                unmatched_detections.remove(best_match_idx)

        # 2. マッチしなかった検出結果を新しいトラックとして追加
        for i in unmatched_detections:
            detection = current_detected_hands[i]
            tracked_hands[next_track_id] = {
                'pos': detection['pos'],
                'landmarks': detection['landmarks'],
                'last_seen': frame_count,
                'deque': deque(maxlen=5000),
                'prev_drawing': False
            }
            next_track_id += 1

        # --- 手ごとのジェスチャー認識と描画 ---
        active_track_ids = list(tracked_hands.keys())
        for track_id in active_track_ids:
            # 3. 長時間見失ったトラックを削除
            if frame_count - tracked_hands[track_id]['last_seen'] > TRACK_LIFETIME:
                del tracked_hands[track_id]
                continue

            track_data = tracked_hands[track_id]
            hand_landmarks = track_data['landmarks']

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            index_is_up = index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            middle_is_up = middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

            drawing = False
            
            # どの手でも色選択・消去できるように変更
            if index_is_up and not middle_is_up:
                cv2.circle(frame, (ix, iy), 15, (255, 255, 0), 3)
                if clear_button["pos"][0] < ix < clear_button["pos"][0] + clear_button["size"][0] and clear_button["pos"][1] < iy < clear_button["pos"][1] + clear_button["size"][1]:
                    for t_id in tracked_hands: tracked_hands[t_id]['deque'].clear()
                else:
                    for r in color_rects:
                        if r["pos"][0] < ix < r["pos"][0] + r["size"][0] and r["pos"][1] < iy < r["pos"][1] + r["size"][1]:
                            draw_color = r["bgr"]
            
            distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
            if distance < 0.05: # 正規化座標での距離判定
                track_data['deque'].appendleft(((ix, iy), draw_color, DRAW_THICKNESS))
                drawing = True

            if track_data['prev_drawing'] and not drawing:
                track_data['deque'].appendleft(None)
            track_data['prev_drawing'] = drawing
            
            # IDを画面に表示
            # cv2.putText(frame, f"ID:{track_id}", (int(track_data['pos'][0] * w), int(track_data['pos'][1] * h)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        # --- UIと軌跡の描画 ---
        for r in color_rects:
            cv2.rectangle(frame, r["pos"], (r["pos"][0] + r["size"][0], r["pos"][1] + r["size"][1]), r["bgr"], -1)
            if r["bgr"] == draw_color: cv2.rectangle(frame, r["pos"], (r["pos"][0] + r["size"][0], r["pos"][1] + r["size"][1]), (255,255,255), 4)
        cv2.rectangle(frame, clear_button["pos"], (clear_button["pos"][0]+clear_button["size"][0], clear_button["pos"][1]+clear_button["size"][1]), (180,180,180), -1)
        cv2.putText(frame, clear_button["text"], (clear_button["pos"][0]+10, clear_button["pos"][1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        for track_data in tracked_hands.values():
            dq = track_data['deque']
            for i in range(1, len(dq)):
                if dq[i - 1] is not None and dq[i] is not None:
                    p1, color1, thick1 = dq[i-1]
                    p2, color2, thick2 = dq[i]
                    cv2.line(frame, p1, p2, color1, thick1)

        cv2.imshow('AR Drawing Game', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    selected_camera_index = select_camera()
    if selected_camera_index is not None:
        ar_drawing_game(selected_camera_index)