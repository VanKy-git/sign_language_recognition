import os
import numpy as np
from scipy import signal
from tensorflow.keras.utils import to_categorical

# --- CẤU HÌNH ---
INPUT_FOLDER = "npy_datas"    
OUTPUT_X = "X_10tu.npy"
OUTPUT_Y = "y_10tu_optimized.npy"
TARGET_FRAMES = 60      
MOTION_THRESHOLD = 0.5  

# --- HÀM 1: CẮT BỎ ĐOẠN ĐẦU/CUỐI (TRIM SILENCE) ---
def trim_sequence(sequence, threshold=MOTION_THRESHOLD):
    diff = np.abs(np.diff(sequence, axis=0))
    motion_score = np.sum(diff, axis=1)
    active_frames = np.where(motion_score > threshold)[0]
    
    if len(active_frames) == 0: return sequence
    
    start_index = active_frames[0]
    end_index = min(active_frames[-1] + 2, len(sequence))
    trimmed = sequence[start_index:end_index]
    
    if len(trimmed) < 5: return sequence
    return trimmed

# --- HÀM 2: ÉP KHUNG HÌNH (RESAMPLE) ---
def resample_sequence(sequence, target_len):
    if len(sequence) == target_len: return sequence
    return signal.resample(sequence, target_len)

# --- QUY TRÌNH CHÍNH (CHỈ LẤY DỮ LIỆU GỐC) ---
def process_data_raw_only():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Lỗi: Không tìm thấy {INPUT_FOLDER}")
        return

    actions = os.listdir(INPUT_FOLDER)
    label_map = {label: num for num, label in enumerate(actions)}
    
    sequences = []
    labels = []
    
    print(f"Bản đồ từ vựng: {label_map}")
    print("Bắt đầu xử lý: Chỉ Trim -> Resample (KHÔNG NHÂN BẢN)...")

    for action in actions:
        action_path = os.path.join(INPUT_FOLDER, action)
        if not os.path.isdir(action_path): continue
        
        file_list = os.listdir(action_path)
        print(f"- Đang xử lý '{action}' ({len(file_list)} videos)...")
        
        for file_name in file_list:
            if not file_name.endswith('.npy'): continue
            
            # Load file gốc
            res = np.load(os.path.join(action_path, file_name))
            
            # BƯỚC 1: TRIM SILENCE 
            trimmed_res = trim_sequence(res)
            
            # BƯỚC 2: RESAMPLE
            final_res = resample_sequence(trimmed_res, TARGET_FRAMES)
            
            # Lưu ĐÚNG 1 BẢN GỐC NÀY VÀO DATASET
            sequences.append(final_res)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    print("\n--- KẾT QUẢ DỮ LIỆU GỐC ---")
    print(f"X shape: {X.shape}") # Sẽ là (150, 60, 258) thay vì 600
    print(f"y shape: {y.shape}") # Sẽ là (150, 3)
    
    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)
    print("Hoàn tất! Dữ liệu đã sạch, KHÔNG chứa Augment.")

if __name__ == "__main__":
    process_data_raw_only()