"""
DEMO: Kiểm tra thuật toán cắt đoạn đầu/cuối không có hành động
Chạy script này để xem nó hoạt động như thế nào trên 1 video
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# THUẬT TOÁN PHÁT HIỆN RANH GIỚI HÀNH ĐỘNG
# =============================================================================

def detect_action_boundaries(keypoints_sequence, threshold=0.01):
    """
    Phát hiện khi nào BẮT ĐẦU và KẾT THÚC hành động
    
    Nguyên lý:
    1. Tính độ thay đổi giữa các frame liên tiếp
    2. Frame nào thay đổi nhiều = đang có hành động
    3. Tìm frame đầu tiên và cuối cùng có chuyển động
    """
    n_frames = len(keypoints_sequence)
    
    # Tính độ chuyển động giữa các frame
    movement = []
    for i in range(1, n_frames):
        # Khoảng cách Euclidean giữa frame hiện tại và frame trước
        diff = np.linalg.norm(keypoints_sequence[i] - keypoints_sequence[i-1])
        movement.append(diff)
    
    movement = np.array(movement)
    
    if movement.max() == 0:
        return 0, n_frames - 1
    
    # Chuẩn hóa về 0-1
    movement_norm = movement / movement.max()
    
    # Tìm các frame có chuyển động vượt ngưỡng
    active_frames = np.where(movement_norm > threshold)[0]
    
    if len(active_frames) == 0:
        return 0, n_frames - 1
    
    # Lùi lại 2 frame và tiến 2 frame để an toàn
    start_frame = max(0, active_frames[0] - 2)
    end_frame = min(n_frames - 1, active_frames[-1] + 2)
    
    return start_frame, end_frame


def smart_extract_frames(keypoints_sequence, target_length=60):
    """
    Trích xuất về target_length frames một cách thông minh
    """
    n_frames = len(keypoints_sequence)
    
    # Bước 1: Cắt bỏ đầu/cuối
    start, end = detect_action_boundaries(keypoints_sequence)
    action_part = keypoints_sequence[start:end+1]
    action_length = len(action_part)
    
    # Bước 2: Resample
    if action_length == 0:
        return np.zeros((target_length, 258))
    elif action_length >= target_length:
        # Downsampling: lấy đều
        indices = np.linspace(0, action_length - 1, target_length, dtype=int)
        return action_part[indices]
    else:
        # Upsampling: lặp lại
        repeat_factor = int(np.ceil(target_length / action_length))
        repeated = np.tile(action_part, (repeat_factor, 1))
        return repeated[:target_length]


# =============================================================================
# DEMO TRỰC QUAN
# =============================================================================

def demo_on_video(video_path):
    """
    Demo thuật toán trên 1 video cụ thể
    """
    print("\n" + "="*70)
    print("🎬 DEMO THUẬT TOÁN CẮT ĐẦU/CUỐI")
    print("="*70)
    
    # Load video
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy: {video_path}")
        return
    
    original = np.load(video_path)
    print(f"\n📹 Video: {os.path.basename(video_path)}")
    print(f"   Số frame gốc: {len(original)}")
    
    # Tính movement
    movement = []
    for i in range(1, len(original)):
        diff = np.linalg.norm(original[i] - original[i-1])
        movement.append(diff)
    
    movement = np.array(movement)
    movement_norm = movement / movement.max() if movement.max() > 0 else movement
    
    # Phát hiện ranh giới
    start, end = detect_action_boundaries(original)
    
    print(f"\n🔍 Phát hiện ranh giới:")
    print(f"   Frame bắt đầu hành động: {start}")
    print(f"   Frame kết thúc hành động: {end}")
    print(f"   Độ dài phần hành động: {end - start + 1} frames")
    print(f"   → Cắt bỏ {start} frames đầu, {len(original) - end - 1} frames cuối")
    
    # Trích xuất về 60 frames
    extracted = smart_extract_frames(original, target_length=60)
    
    print(f"\n✂️ Sau khi trích xuất:")
    print(f"   Số frame cuối cùng: {len(extracted)}")
    print(f"   Shape: {extracted.shape}")
    
    # VẼ BIỂU ĐỒ
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Movement của video gốc
    axes[0].plot(movement_norm, linewidth=1.5, color='blue')
    axes[0].axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Threshold=0.01')
    axes[0].axvspan(start, end, alpha=0.2, color='green', label='Phần giữ lại')
    
    # Đánh dấu phần bị cắt
    if start > 0:
        axes[0].axvspan(0, start, alpha=0.2, color='red', label='Cắt đầu')
    if end < len(original) - 1:
        axes[0].axvspan(end, len(original) - 1, alpha=0.2, color='red', label='Cắt cuối')
    
    axes[0].set_title(f'Video gốc: {len(original)} frames', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Độ chuyển động (normalized)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Keypoints của một vài điểm quan trọng (tay phải)
    # Lấy 3 điểm của tay phải: cổ tay, đầu ngón giữa, đầu ngón út
    right_hand_start = 195
    wrist_x = original[:, right_hand_start + 0]  # x của cổ tay phải
    middle_x = original[:, right_hand_start + 12]  # x của đầu ngón giữa
    pinky_x = original[:, right_hand_start + 18]  # x của đầu ngón út
    
    axes[1].plot(wrist_x, label='Cổ tay X', alpha=0.7)
    axes[1].plot(middle_x, label='Ngón giữa X', alpha=0.7)
    axes[1].plot(pinky_x, label='Ngón út X', alpha=0.7)
    axes[1].axvspan(start, end, alpha=0.2, color='green')
    
    axes[1].set_title('Vị trí một vài điểm tay phải theo thời gian', fontsize=12)
    axes[1].set_ylabel('Toạ độ X')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Movement của video sau khi trích xuất
    ext_movement = []
    for i in range(1, len(extracted)):
        diff = np.linalg.norm(extracted[i] - extracted[i-1])
        ext_movement.append(diff)
    
    ext_movement = np.array(ext_movement)
    ext_movement_norm = ext_movement / ext_movement.max() if ext_movement.max() > 0 else ext_movement
    
    axes[2].plot(ext_movement_norm, linewidth=1.5, color='green')
    axes[2].set_title(f'Video sau xử lý: {len(extracted)} frames (cố định)', 
                     fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Độ chuyển động (normalized)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Lưu hình
    output_name = f"demo_{os.path.basename(video_path).replace('.npy', '')}.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\n💾 Đã lưu biểu đồ: {output_name}")
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "🎯"*35)
    print("     DEMO THUẬT TOÁN CẮT ĐOẠN ĐẦU/CUỐI")
    print("🎯"*35)
    
    # Hướng dẫn
    print("\nCách dùng:")
    print("1. Sửa VIDEO_PATH thành đường dẫn file .npy của bạn")
    print("2. Chạy script này")
    print("3. Xem biểu đồ để hiểu thuật toán hoạt động thế nào")
    
    # Ví dụ: Demo với video đầu tiên của action 'hello'
    VIDEO_PATH = 'keypoint_data/hello/0.npy'  # ← SỬA ĐƯỜNG DẪN NÀY
    
    # Hoặc tự động tìm file đầu tiên
    if not os.path.exists(VIDEO_PATH):
        print(f"\n⚠️ Không tìm thấy: {VIDEO_PATH}")
        print("   Đang tìm video khác...")
        
        for action in ['hello', 'book', 'drink']:
            folder = f'keypoint_data/{action}'
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.endswith('.npy')]
                if files:
                    VIDEO_PATH = os.path.join(folder, files[0])
                    print(f"   → Tìm thấy: {VIDEO_PATH}")
                    break
    
    if os.path.exists(VIDEO_PATH):
        demo_on_video(VIDEO_PATH)
    else:
        print(f"\n❌ Không tìm thấy video nào để demo!")
        print(f"   Hãy chạy script trích xuất keypoints trước.")
    
    print("\n" + "="*70)
    print("💡 GIẢI THÍCH:")
    print("="*70)
    print("""
Biểu đồ có 3 phần:

1️⃣ Video gốc (trên cùng):
   - Đường xanh: Độ chuyển động giữa các frame
   - Vùng xanh lá: Phần có hành động (giữ lại)
   - Vùng đỏ: Phần không có hành động (bị cắt bỏ)
   
2️⃣ Keypoints của tay phải (giữa):
   - Xem toạ độ 3 điểm tay thay đổi thế nào theo thời gian
   - Phần đầu/cuối thường ít thay đổi → bị cắt
   
3️⃣ Video sau xử lý (dưới cùng):
   - Đã cắt bỏ phần đầu/cuối
   - Trích xuất đều về 60 frames
   - Giữ nguyên động tác chính
   
🎯 Mục tiêu:
   - Loại bỏ phần "đứng yên" đầu/cuối
   - Chỉ giữ lại phần "đang làm hành động"
   - Chuẩn hóa về 60 frames để dễ train model
""")
    print("="*70)


if __name__ == "__main__":
    main()
