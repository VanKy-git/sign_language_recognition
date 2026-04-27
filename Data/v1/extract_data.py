import cv2
import numpy as np
import mediapipe as mp
import os

# Import lõi xử lý đặc trưng vừa tạo
from feature_engineering import process_single_video_features

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "raw_videos")  # Thư mục chứa video gốc
OUTPUT_FOLDER = os.path.join(BASE_DIR, "npy_datas")  # Thư mục chứa file .npy đã qua xử lý đặc trưng
TARGET_QTY = 50                    # Số lượng video tối đa mỗi từ

# --- KHỞI TẠO MEDIAPIPE ---
mp_holistic = mp.solutions.holistic

def extract_raw_keypoints(results):
    """
    Trích xuất đúng 162 giá trị thô: Pose thân trên (36) + Tay trái (63) + Tay phải (63)
    Không thực hiện chuẩn hóa ở bước này.
    """
    # 1. Pose Thân trên (12 điểm: từ index 11 đến 22 của MediaPipe)
    # Bao gồm: 2 vai, 2 khuỷu tay, 2 cổ tay, và các đốt ngón tay cơ sở. Bỏ qua mặt và chân.
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[11:23]]).flatten()
    else:
        pose = np.zeros(12 * 3)
        
    # 2. Left Hand (Lấy tọa độ thô tuyệt đối)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)
        
    # 3. Right Hand (Lấy tọa độ thô tuyệt đối)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)
        
    return np.concatenate([pose, lh, rh])

def process_and_save():
    if not os.path.exists(INPUT_FOLDER):
        print(f"LỖI: Không tìm thấy thư mục '{INPUT_FOLDER}'.")
        return

    actions = os.listdir(INPUT_FOLDER)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for action in actions:
            action_path_in = os.path.join(INPUT_FOLDER, action)
            action_path_out = os.path.join(OUTPUT_FOLDER, action)
            
            if not os.path.isdir(action_path_in): continue
            
            # Bỏ qua nếu đã extract đủ
            if os.path.exists(action_path_out) and len(os.listdir(action_path_out)) > 0:
                print(f"Bỏ qua '{action}': Đã có sẵn dữ liệu.")
                continue
            
            os.makedirs(action_path_out, exist_ok=True)
            video_files = os.listdir(action_path_in)
            saved_count = 0
            
            print(f"--- Đang xử lý: '{action}' ---")
            
            for video_file in video_files:
                if saved_count >= TARGET_QTY:
                    break
                    
                video_path = os.path.join(action_path_in, video_file)
                cap = cv2.VideoCapture(video_path)
                
                raw_frames_data = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Tối ưu: chuyển sang RGB để MediaPipe đọc
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    
                    # Trích xuất dữ liệu THÔ
                    keypoints = extract_raw_keypoints(results)
                    raw_frames_data.append(keypoints)
                    
                cap.release()
                
                # --- ĐƯA QUA LÕI XỬ LÝ ĐẶC TRƯNG ---
                if len(raw_frames_data) > 10:
                    # Gọi pipeline từ feature_engineering.py
                    processed_features = process_single_video_features(raw_frames_data)
                    
                    if processed_features is not None:
                        npy_path = os.path.join(action_path_out, f"{action}_{saved_count}.npy")
                        # Lưu array đã qua xử lý. Shape lúc này sẽ là (50, 324) 
                        # 324 = 162 tọa độ gốc + 162 vector vận tốc
                        np.save(npy_path, processed_features)
                        saved_count += 1
                        print(f"  + Đã lưu: {action}_{saved_count}.npy (Shape: {processed_features.shape})")
                else:
                    print(f"  ! Bỏ qua: {video_file} (Video quá ngắn)")

            print(f"-> Hoàn tất '{action}'. Tổng: {saved_count}/{TARGET_QTY} video.")

if __name__ == "__main__":
    process_and_save()
    print("\n=== ĐÃ XỬ LÝ XONG TOÀN BỘ DỮ LIỆU THÀNH ĐẶC TRƯNG ===")