import cv2
import numpy as np
import mediapipe as mp
import os

from featurev2 import process_single_video_features

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)
INPUT_FOLDER = os.path.join(DATA_DIR, "raw_videos")
OUTPUT_FOLDER = os.path.join(DATA_DIR, "npy_datav2")
TARGET_QTY = 50                    

mp_holistic = mp.solutions.holistic

# 20 điểm khuôn mặt (Môi, lông mày, mắt)
FACE_INDICES = [
    0, 13, 14, 17, 61, 291, 39, 181, 269, 405,  
    46, 52, 65, 276, 282, 295, 33, 133, 362, 263                           
]

def extract_raw_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[11:23]]).flatten()
    else: pose = np.zeros(12 * 3)
        
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else: lh = np.zeros(21 * 3)
        
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else: rh = np.zeros(21 * 3)
        
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[i].x, 
                          results.face_landmarks.landmark[i].y, 
                          results.face_landmarks.landmark[i].z] 
                         for i in FACE_INDICES]).flatten()
    else: face = np.zeros(len(FACE_INDICES) * 3)
        
    # Tổng: 36 + 63 + 63 + 60 = 222
    return np.concatenate([pose, lh, rh, face])

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
            
            os.makedirs(action_path_out, exist_ok=True)
            video_files = os.listdir(action_path_in)
            saved_count = len(os.listdir(action_path_out))
            
            if saved_count >= TARGET_QTY:
                print(f"Bỏ qua '{action}': Đã có đủ dữ liệu.")
                continue
                
            print(f"--- Đang xử lý: '{action}' ---")
            for video_file in video_files:
                if saved_count >= TARGET_QTY: break
                
                # Check nếu file đã được extract trước đó thì bỏ qua để tiết kiệm thời gian
                npy_path = os.path.join(action_path_out, f"{action}_{saved_count}.npy")
                if os.path.exists(npy_path):
                    saved_count += 1
                    continue
                    
                video_path = os.path.join(action_path_in, video_file)
                cap = cv2.VideoCapture(video_path)
                raw_frames_data = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    
                    keypoints = extract_raw_keypoints(results)
                    raw_frames_data.append(keypoints)
                    
                cap.release()
                
                if len(raw_frames_data) > 10:
                    processed_features = process_single_video_features(raw_frames_data)
                    if processed_features is not None:
                        np.save(npy_path, processed_features)
                        saved_count += 1
                        print(f"  + Đã lưu: {action}_{saved_count}.npy (Shape: {processed_features.shape})")
                else:
                    print(f"  ! Bỏ qua: {video_file} (Video quá ngắn)")

            print(f"-> Hoàn tất '{action}'. Tổng: {saved_count}/{TARGET_QTY} video.")

if __name__ == "__main__":
    process_and_save()
    print("\n=== ĐÃ XỬ LÝ XONG TOÀN BỘ DỮ LIỆU THÀNH ĐẶC TRƯNG ===")