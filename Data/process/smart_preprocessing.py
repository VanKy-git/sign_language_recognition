"""
TIỀN XỬ LÝ DỮ LIỆU - PHIÊN BẢN TỐI ƯU
- Chuẩn hóa: 50 videos/từ
- Cố định: 60 frames/video
- Thông minh: Tự động cắt bỏ đoạn đầu/cuối không có hành động
"""

import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# =============================================================================
# CẤU HÌNH
# =============================================================================

RAW_VIDEO_PATH = 'Data/raw_videos'
KEYPOINT_PATH = 'keypoint_data'
FINAL_PATH = 'final_data'

ACTIONS = ['book', 'hello', 'drink']
TARGET_VIDEOS_PER_ACTION = 50  # Cố định 50 videos/từ
FIXED_LENGTH = 60              # Cố định 60 frames

# =============================================================================
# HÀM HỖ TRỢ
# =============================================================================

def detect_action_boundaries(keypoints_sequence, threshold=0.01):
    """
    Phát hiện khi nào bắt đầu và kết thúc hành động
    
    Ý tưởng: 
    - Khi chưa làm gì: keypoints ít thay đổi (người đứng yên)
    - Khi làm hành động: keypoints thay đổi nhiều (tay cử động)
    - Sau khi làm xong: lại ít thay đổi
    
    Args:
        keypoints_sequence: mảng (n_frames, 258)
        threshold: ngưỡng chuyển động để coi là "đang làm hành động"
    
    Returns:
        start_frame, end_frame: vị trí bắt đầu và kết thúc hành động
    """
    n_frames = len(keypoints_sequence)
    
    # Tính độ thay đổi giữa các frame liên tiếp
    movement = []
    for i in range(1, n_frames):
        # Tính khoảng cách Euclidean giữa frame i và frame i-1
        diff = np.linalg.norm(keypoints_sequence[i] - keypoints_sequence[i-1])
        movement.append(diff)
    
    movement = np.array(movement)
    
    # Nếu toàn bộ video không có chuyển động → trả về toàn bộ
    if movement.max() == 0:
        return 0, n_frames - 1
    
    # Normalize movement về 0-1
    movement_norm = movement / movement.max()
    
    # Tìm các frame có chuyển động lớn
    active_frames = np.where(movement_norm > threshold)[0]
    
    if len(active_frames) == 0:
        # Không có frame nào vượt ngưỡng → trả về toàn bộ
        return 0, n_frames - 1
    
    # Frame bắt đầu: frame đầu tiên có chuyển động
    start_frame = max(0, active_frames[0] - 2)  # Lùi lại 2 frame cho chắc
    
    # Frame kết thúc: frame cuối cùng có chuyển động
    end_frame = min(n_frames - 1, active_frames[-1] + 2)  # Tiến 2 frame
    
    return start_frame, end_frame


def smart_extract_frames(keypoints_sequence, target_length=60):
    """
    Trích xuất THÔNG MINH về target_length frames
    
    Quy trình:
    1. Cắt bỏ đoạn đầu/cuối không có hành động
    2. Từ phần còn lại, lấy đều về target_length frames
    
    Args:
        keypoints_sequence: mảng (n_frames, 258)
        target_length: số frame mục tiêu (60)
    
    Returns:
        mảng (target_length, 258)
    """
    n_frames = len(keypoints_sequence)
    
    # BƯỚC 1: Phát hiện ranh giới hành động
    start, end = detect_action_boundaries(keypoints_sequence)
    
    # Cắt lấy phần có hành động
    action_part = keypoints_sequence[start:end+1]
    action_length = len(action_part)
    
    # BƯỚC 2: Resample về target_length
    if action_length == 0:
        # Trường hợp đặc biệt: video toàn 0
        return np.zeros((target_length, 258))
    
    elif action_length >= target_length:
        # Video dài → lấy đều (downsampling)
        indices = np.linspace(0, action_length - 1, target_length, dtype=int)
        return action_part[indices]
    
    else:
        # Video ngắn → lặp lại (upsampling)
        # Cách 1: Lặp lại frame (đơn giản)
        repeat_factor = int(np.ceil(target_length / action_length))
        repeated = np.tile(action_part, (repeat_factor, 1))
        return repeated[:target_length]


def visualize_extraction(original, extracted, video_name):
    """
    Vẽ biểu đồ để xem quá trình cắt/trích xuất
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Tính movement cho original
    orig_movement = []
    for i in range(1, len(original)):
        diff = np.linalg.norm(original[i] - original[i-1])
        orig_movement.append(diff)
    
    # Tính movement cho extracted
    ext_movement = []
    for i in range(1, len(extracted)):
        diff = np.linalg.norm(extracted[i] - extracted[i-1])
        ext_movement.append(diff)
    
    # Plot original
    axes[0].plot(orig_movement, linewidth=1.5)
    axes[0].set_title(f'Original: {len(original)} frames')
    axes[0].set_ylabel('Movement')
    axes[0].grid(True, alpha=0.3)
    
    # Highlight phần được giữ lại
    start, end = detect_action_boundaries(original)
    axes[0].axvspan(start, end, alpha=0.2, color='green', label='Kept')
    axes[0].legend()
    
    # Plot extracted
    axes[1].plot(ext_movement, linewidth=1.5, color='green')
    axes[1].set_title(f'Extracted: {len(extracted)} frames')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Movement')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'debug_{video_name}.png', dpi=100)
    plt.close()


# =============================================================================
# BƯỚC 1: CHỌN 50 VIDEO TỐT NHẤT CHO MỖI TỪ
# =============================================================================

def select_best_videos(action, n_videos=50):
    """
    Chọn n_videos tốt nhất từ folder keypoints
    
    Tiêu chí:
    1. Có đủ frames (>= 20)
    2. Không có quá nhiều frame bị mất detection (< 30%)
    3. Ưu tiên video có độ dài gần 60 frames
    """
    folder = os.path.join(KEYPOINT_PATH, action)
    
    if not os.path.exists(folder):
        print(f"❌ Không tìm thấy: {folder}")
        return []
    
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    
    # Đánh giá từng video
    video_scores = []
    
    for file in files:
        path = os.path.join(folder, file)
        data = np.load(path)
        
        # Tiêu chí 1: Số frames
        n_frames = len(data)
        if n_frames < 20:
            continue  # Bỏ qua video quá ngắn
        
        # Tiêu chí 2: Tỉ lệ frame bị mất detection
        zero_frames = np.where(data.sum(axis=1) == 0)[0]
        zero_ratio = len(zero_frames) / n_frames
        
        if zero_ratio > 0.3:
            continue  # Bỏ qua video có quá nhiều frame lỗi
        
        # Tiêu chí 3: Độ gần với 60 frames
        distance_to_target = abs(n_frames - 60)
        
        # Tính điểm tổng hợp (càng cao càng tốt)
        score = (
            100 - zero_ratio * 100  # Ít lỗi hơn = tốt hơn
            - distance_to_target * 0.5  # Gần 60 frames = tốt hơn
        )
        
        video_scores.append({
            'file': file,
            'score': score,
            'frames': n_frames,
            'zero_ratio': zero_ratio
        })
    
    # Sắp xếp theo điểm
    video_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Lấy top n_videos
    selected = video_scores[:n_videos]
    
    print(f"\n📁 {action.upper()}: Đã chọn {len(selected)}/{len(files)} videos tốt nhất")
    if selected:
        print(f"   Điểm cao nhất: {selected[0]['score']:.1f} ({selected[0]['file']})")
        print(f"   Điểm thấp nhất: {selected[-1]['score']:.1f} ({selected[-1]['file']})")
    
    return [v['file'] for v in selected]


# =============================================================================
# BƯỚC 2: XỬ LÝ VÀ CHUẨN HÓA
# =============================================================================

def process_all_data(visualize_samples=False):
    """
    Xử lý toàn bộ dữ liệu:
    1. Chọn 50 videos/từ
    2. Trích xuất 60 frames thông minh
    3. Normalize
    4. Chia train/val/test
    """
    
    print("\n" + "="*70)
    print("🚀 BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
    print("="*70)
    
    all_data = []
    all_labels = []
    label_map = {'book': 0, 'hello': 1, 'drink': 2}
    
    # BƯỚC 1: Chọn và xử lý từng action
    for action in ACTIONS:
        print(f"\n{'='*70}")
        print(f"📊 Xử lý: {action.upper()}")
        print(f"{'='*70}")
        
        # Chọn 50 videos tốt nhất
        selected_files = select_best_videos(action, TARGET_VIDEOS_PER_ACTION)
        
        if len(selected_files) < TARGET_VIDEOS_PER_ACTION:
            print(f"⚠️ Chỉ có {len(selected_files)}/{TARGET_VIDEOS_PER_ACTION} videos đủ tốt!")
            if len(selected_files) < 30:
                print(f"❌ Quá ít! Cần ít nhất 30 videos. Bỏ qua action này.")
                continue
        
        # Xử lý từng video
        folder = os.path.join(KEYPOINT_PATH, action)
        
        for i, file in enumerate(selected_files):
            # Load keypoints
            original = np.load(os.path.join(folder, file))
            
            # Trích xuất thông minh về 60 frames
            extracted = smart_extract_frames(original, FIXED_LENGTH)
            
            # Lưu vào danh sách
            all_data.append(extracted)
            all_labels.append(label_map[action])
            
            # Visualize một vài mẫu để debug
            if visualize_samples and i < 2:
                visualize_extraction(original, extracted, f"{action}_{i}")
            
            # In tiến trình
            if (i + 1) % 10 == 0:
                print(f"   Đã xử lý: {i+1}/{len(selected_files)}")
        
        print(f"   ✅ Hoàn thành: {len(selected_files)} videos → (60, 258)")
    
    # Chuyển thành numpy arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    print(f"\n{'='*70}")
    print(f"📊 TỔNG KẾT DỮ LIỆU")
    print(f"{'='*70}")
    print(f"   Tổng số videos: {len(all_data)}")
    print(f"   Shape: {all_data.shape}")
    print(f"   Phân bố class:")
    for action, label in label_map.items():
        count = np.sum(all_labels == label)
        print(f"      {action}: {count} videos")
    
    # BƯỚC 2: Normalize
    print(f"\n{'='*70}")
    print(f"🔧 NORMALIZE DỮ LIỆU")
    print(f"{'='*70}")
    
    # Flatten để fit scaler
    all_data_flat = all_data.reshape(-1, 258)
    
    scaler = StandardScaler()
    scaler.fit(all_data_flat)
    
    # Transform
    normalized_flat = scaler.transform(all_data_flat)
    normalized = normalized_flat.reshape(all_data.shape)
    
    # Lưu scaler
    os.makedirs(FINAL_PATH, exist_ok=True)
    joblib.dump(scaler, os.path.join(FINAL_PATH, 'scaler.pkl'))
    
    print(f"   ✅ Đã normalize và lưu scaler.pkl")
    print(f"   Mean: {scaler.mean_[:5].round(3)}")
    print(f"   Std:  {scaler.scale_[:5].round(3)}")
    
    # BƯỚC 3: Chia train/val/test
    print(f"\n{'='*70}")
    print(f"📂 CHIA TRAIN/VAL/TEST")
    print(f"{'='*70}")
    
    # Shuffle
    indices = np.arange(len(normalized))
    np.random.seed(42)
    np.random.shuffle(indices)
    normalized = normalized[indices]
    all_labels = all_labels[indices]
    
    # Chia 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        normalized, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Lưu
    np.save(os.path.join(FINAL_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(FINAL_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(FINAL_PATH, 'X_val.npy'), X_val)
    np.save(os.path.join(FINAL_PATH, 'y_val.npy'), y_val)
    np.save(os.path.join(FINAL_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(FINAL_PATH, 'y_test.npy'), y_test)
    np.save(os.path.join(FINAL_PATH, 'label_map.npy'), label_map)
    
    print(f"\n   ✅ Đã lưu vào: {FINAL_PATH}/")
    print(f"\n   📊 Phân bố:")
    print(f"      Train: {len(X_train)} samples ({len(X_train)/len(normalized)*100:.1f}%)")
    print(f"      Val:   {len(X_val)} samples ({len(X_val)/len(normalized)*100:.1f}%)")
    print(f"      Test:  {len(X_test)} samples ({len(X_test)/len(normalized)*100:.1f}%)")
    
    print(f"\n   📊 Phân bố class trong từng split:")
    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        counts = [np.sum(y_split == label) for label in range(3)]
        print(f"      {split_name}: book={counts[0]}, hello={counts[1]}, drink={counts[2]}")
    
    # BƯỚC 4: Tạo file thông tin
    info = f"""
# THÔNG TIN DATASET

## Cấu hình
- Số videos/từ: {TARGET_VIDEOS_PER_ACTION}
- Số frames/video: {FIXED_LENGTH}
- Tổng số videos: {len(all_data)}

## Shape dữ liệu
- X_train: {X_train.shape}
- X_val: {X_val.shape}
- X_test: {X_test.shape}

## Label mapping
- book: 0
- hello: 1
- drink: 2

## Phân bố
- Train: {len(X_train)} ({len(X_train)/len(normalized)*100:.1f}%)
- Val: {len(X_val)} ({len(X_val)/len(normalized)*100:.1f}%)
- Test: {len(X_test)} ({len(X_test)/len(normalized)*100:.1f}%)

## Phương pháp tiền xử lý
1. Chọn {TARGET_VIDEOS_PER_ACTION} videos tốt nhất cho mỗi từ
2. Phát hiện ranh giới hành động (cắt bỏ đoạn đầu/cuối không có động tác)
3. Trích xuất đều về {FIXED_LENGTH} frames
4. Normalize với StandardScaler (mean=0, std=1)
5. Chia train/val/test với stratify

## Files
- scaler.pkl: StandardScaler để normalize dữ liệu mới
- label_map.npy: Mapping từ tên class sang số
- X_train.npy, y_train.npy: Training set
- X_val.npy, y_val.npy: Validation set
- X_test.npy, y_test.npy: Test set
"""
    
    with open(os.path.join(FINAL_PATH, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(info)
    
    print(f"\n   💾 Đã tạo README.md")
    
    print(f"\n{'🎉'*35}")
    print(f"     HOÀN TẤT TIỀN XỬ LÝ!")
    print(f"{'🎉'*35}")
    print(f"\n📂 Tất cả dữ liệu đã sẵn sàng tại: {FINAL_PATH}/")
    print(f"   → Có thể bắt đầu train model ngay!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Tiền xử lý dữ liệu Sign Language')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize 2 video đầu của mỗi action để debug')
    
    args = parser.parse_args()
    
    # Kiểm tra folder keypoint_data có tồn tại không
    if not os.path.exists(KEYPOINT_PATH):
        print(f"❌ Không tìm thấy folder: {KEYPOINT_PATH}")
        print(f"   → Chạy script trích xuất keypoints trước!")
        return
    
    # Chạy xử lý
    process_all_data(visualize_samples=args.visualize)


if __name__ == "__main__":
    main()
