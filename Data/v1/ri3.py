import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import os
from collections import deque, Counter
from tensorflow.keras.models import load_model

# Import lõi đặc trưng đã viết
from feature_engineering import process_single_video_features

# ============= CẤU HÌNH =============
def load_actions_from_folders():
    """Tự động lấy danh sách lớp từ thư mục dữ liệu."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        os.path.join(base_dir, 'npy_datas'),
        os.path.join(base_dir, 'old_npydata'),
        os.path.join(base_dir, 'raw_videos'),
    ]

    data_dir = next((d for d in candidate_dirs if os.path.isdir(d)), None)
    if data_dir is None:
        raise FileNotFoundError("Không tìm thấy thư mục dữ liệu để suy ra ACTIONS.")

    actions = sorted(
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith('.')
    )

    if not actions:
        raise ValueError(f"Không có thư mục lớp nào trong: {data_dir}")

    return actions


ACTIONS = load_actions_from_folders()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'hybrid_model_50tu.keras')
MEAN_PATH = os.path.join(BASE_DIR, 'model', 'train_mean.npy')
STD_PATH = os.path.join(BASE_DIR, 'model', 'train_std.npy')

ESP32_URL = "http://172.31.99.127:81/stream"  # Bật dòng này nếu dùng ESP32
# ESP32_URL = 0  # Dùng 0 cho Webcam máy tính để test trước

CONFIDENCE_THRESHOLD = 0.5

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ============= THREAD XỬ LÝ LAG WIFI =============
class VideoStreamThread:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()

# ============= CÁC HÀM XỬ LÝ ĐẶC TRƯNG MỚI =============
def extract_raw_keypoints(results):
    """Cào đúng 162 giá trị thô (Pose thân trên + 2 tay)"""
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[11:23]]).flatten()
    else:
        pose = np.zeros(12*3)
        
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)
        
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
        
    return np.concatenate([pose, lh, rh])

def calculate_motion_score(buffer):
    """Tính vận tốc dựa trên sự di chuyển của bàn tay"""
    if len(buffer) < 2: return 0.0
    seq = np.array(buffer)
    hands_only = seq[:, 36:] # Chỉ tính chuyển động tay
    diffs = np.abs(np.diff(hands_only, axis=0))
    return np.mean(np.sum(diffs, axis=1))

# ============= VẼ UI SIDEBAR =============
def draw_sidebar(info):
    sidebar = np.zeros((720, 400, 3), dtype=np.uint8)
    sidebar[:] = (30, 30, 30) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(sidebar, "SYSTEM STATUS", (20, 40), font, 0.8, (255, 255, 255), 2)
    cv2.line(sidebar, (20, 50), (380, 50), (100, 100, 100), 1)
    
    fps_color = (0, 255, 0) if info['fps'] > 15 else (0, 0, 255)
    cv2.putText(sidebar, f"FPS: {info['fps']:.1f}", (20, 90), font, 0.7, fps_color, 2)
    
    cv2.putText(sidebar, f"Motion: {info['motion']:.3f}", (200, 90), font, 0.7, (0, 255, 255), 2)
    
    cv2.putText(sidebar, "CURRENT PREDICTION", (20, 180), font, 0.8, (255, 255, 255), 2)
    cv2.line(sidebar, (20, 190), (380, 190), (100, 100, 100), 1)
    
    if info['current_action'] == "Recording...":
        cv2.rectangle(sidebar, (20, 220), (380, 340), (0, 0, 150), -1) # Đỏ thẫm
        cv2.putText(sidebar, "Recording...", (40, 290), font, 1.2, (200, 200, 255), 2)
    elif info['current_action']:
        cv2.rectangle(sidebar, (20, 220), (380, 340), (0, 120, 0), -1) # Xanh lá
        cv2.rectangle(sidebar, (20, 220), (380, 340), (0, 255, 0), 2)  
        cv2.putText(sidebar, str(info['current_action']), (40, 280), font, 1.8, (255, 255, 255), 3)
        cv2.putText(sidebar, f"Conf: {info['confidence']*100:.1f}%", (40, 320), font, 0.8, (200, 255, 200), 2)
    else:
        cv2.rectangle(sidebar, (20, 220), (380, 340), (50, 50, 50), -1)
        cv2.putText(sidebar, "Waiting...", (40, 290), font, 1.2, (150, 150, 150), 2)

    cv2.putText(sidebar, "SENTENCE HISTORY", (20, 420), font, 0.8, (255, 255, 255), 2)
    cv2.line(sidebar, (20, 430), (380, 430), (100, 100, 100), 1)
    
    y = 470
    for word in info['sentence'][::-1]:
        cv2.putText(sidebar, f"> {word}", (20, y), font, 1.0, (200, 200, 255), 2)
        y += 40
        
    cv2.putText(sidebar, "Press 'Q' to quit | 'R' to reset", (20, 680), font, 0.6, (100, 100, 100), 1)
    
    return sidebar

# ============= CHƯƠNG TRÌNH CHÍNH =============
def run_realtime_esp32():
    print("1. Loading Model and Normalization Configs...")
    model = load_model(MODEL_PATH)
    train_mean = np.load(MEAN_PATH)
    train_std = np.load(STD_PATH)
    
    print(f"2. Connecting to Camera: {ESP32_URL} ...")
    stream_thread = VideoStreamThread(ESP32_URL).start()
    time.sleep(2.0)
    
    if stream_thread.frame is None:
        print("LỖI: Không nhận được hình ảnh. Kiểm tra lại WiFi/IP!")
        stream_thread.stop()
        return
        
    print("Connected successfully! Hãy thực hiện cử chỉ...")

    # BIẾN CHO CƠ CHẾ START-STOP
    motion_buffer = deque(maxlen=5) 
    action_frames = []              
    is_recording = False            
    
    START_THRESHOLD = 0.04          # Ngưỡng bắt đầu chuyển động
    STOP_THRESHOLD = 0.02           # Ngưỡng kết thúc chuyển động
    MIN_ACTION_FRAMES = 15          # Bỏ qua nếu cử động quá nhanh (<15 frames)
    
    fps_counter = deque(maxlen=30)
    sentence = []
    current_action = None
    display_confidence = 0.0
    prev_time = time.time()
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            frame = stream_thread.read()
            if frame is None: continue
            try: frame = cv2.resize(frame, (960, 720))
            except: continue
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 1. Trích xuất đặc trưng THÔ
            raw_keypoints = extract_raw_keypoints(results)
            motion_buffer.append(raw_keypoints)
            current_motion = calculate_motion_score(list(motion_buffer)) if len(motion_buffer) == 5 else 0
            
            # ================= CƠ CHẾ START-STOP =================
            if not is_recording:
                # Nếu vận tốc vượt ngưỡng -> Bắt đầu ghi
                if current_motion > START_THRESHOLD:
                    is_recording = True
                    action_frames = list(motion_buffer)
                    print("\n[REC] Phát hiện cử động...", end='\r')
            else:
                # Đang ghi -> Lưu frame và vẽ viền đỏ báo hiệu
                action_frames.append(raw_keypoints)
                cv2.rectangle(image, (5, 5), (955, 715), (0, 0, 255), 4)
                
                # Chốt hành động khi tay dừng lại hoặc quá lâu (100 frame)
                if current_motion < STOP_THRESHOLD or len(action_frames) > 100:
                    is_recording = False
                    print(f"\n[STOP] Đã chốt hành động ({len(action_frames)} frames). Đang dự đoán AI...")
                    
                    if len(action_frames) >= MIN_ACTION_FRAMES:
                        # Bơm cả cụm frame vào Lõi xử lý
                        features = process_single_video_features(action_frames)
                        
                        if features is not None:
                            # Chuẩn hóa & Dự đoán
                            features_norm = (features - train_mean) / (train_std + 1e-7)
                            X_input = np.expand_dims(features_norm, axis=0)
                            
                            predictions = model.predict(X_input, verbose=0)[0]
                            best_idx = np.argmax(predictions)
                            best_conf = predictions[best_idx]
                            
                            # Log ra Terminal để theo dõi Top 3
                            top_3_idx = np.argsort(predictions)[-3:][::-1]
                            top_3 = [(ACTIONS[i], predictions[i]) for i in top_3_idx]
                            print(f">>> Result: {ACTIONS[best_idx]} ({best_conf*100:.1f}%) | Top 3: {top_3}")
                            
                            if best_conf > CONFIDENCE_THRESHOLD:
                                current_action = ACTIONS[best_idx]
                                display_confidence = best_conf
                                
                                # Tránh lặp 1 từ liên tiếp
                                if len(sentence) == 0 or sentence[-1] != current_action:
                                    sentence.append(current_action)
                                    if len(sentence) > 5:
                                        sentence = sentence[-5:]
                    else:
                        print(f"-> Bỏ qua vì cử động quá ngắn.")
                    
                    action_frames = [] # Reset để đón từ tiếp theo
            # =====================================================

            # 4. Vẽ Landmarks (Chỉ hiện 12 điểm thân trên và 2 tay)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            if results.pose_landmarks:
                h, w, c = image.shape
                upper_body_connections = [
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22)
                ]
                for start_idx, end_idx in upper_body_connections:
                    start_lm = results.pose_landmarks.landmark[start_idx]
                    end_lm = results.pose_landmarks.landmark[end_idx]
                    start_pos = (int(start_lm.x * w), int(start_lm.y * h))
                    end_pos = (int(end_lm.x * w), int(end_lm.y * h))
                    cv2.line(image, start_pos, end_pos, (0, 255, 255), 2) 
                
                for idx in range(11, 23):
                    landmark = results.pose_landmarks.landmark[idx]
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

            # 5. Tính FPS & Ghép UI
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
            prev_time = curr_time
            fps_counter.append(fps)
            
            ui_info = {
                'fps': np.mean(fps_counter),
                'motion': current_motion,
                'current_action': current_action if not is_recording else "Recording...",
                'confidence': display_confidence if not is_recording else 0.0,
                'sentence': sentence
            }
            
            sidebar = draw_sidebar(ui_info)
            final_app_view = np.hstack((image, sidebar))
            
            cv2.imshow("Sign Language Recognition", final_app_view)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                sentence.clear()
                current_action = None

    stream_thread.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_esp32()