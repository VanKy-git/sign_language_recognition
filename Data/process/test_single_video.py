import cv2
import numpy as np
import mediapipe as mp

# =============================================================================
# GIẢI THÍCH BẢN CHẤT:
# MediaPipe Holistic sẽ phát hiện:
# - Pose (cơ thể): 33 điểm x 4 giá trị (x, y, z, visibility) = 132 số
# - Left Hand (tay trái): 21 điểm x 3 giá trị (x, y, z) = 63 số
# - Right Hand (tay phải): 21 điểm x 3 giá trị (x, y, z) = 63 số
# => Tổng mỗi frame: 132 + 63 + 63 = 258 số (1 mảng dài)
# =============================================================================

# --- CẤU HÌNH ĐƠN GIẢN ---
VIDEO_PATH = 'Data/raw_videos/hello/WIN_20260128_10_47_35_Pro.mp4'  # Đường dẫn video bạn muốn test
OUTPUT_FILE = 'test_output.npy'               # File .npy sẽ lưu

# Khởi tạo MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# --- HÀM 1: PHÁT HIỆN KEYPOINTS TRONG 1 FRAME ---
def mediapipe_detection(image, model):
    """
    Nhận vào: 
    - image: 1 frame ảnh từ video (dạng BGR của OpenCV)
    - model: mô hình MediaPipe Holistic
    
    Xử lý:
    1. Chuyển BGR -> RGB (vì MediaPipe chỉ hiểu RGB)
    2. Khóa ảnh (writeable=False) để tăng tốc độ xử lý
    3. Cho ảnh qua model để phát hiện các điểm khớp
    4. Mở khóa và chuyển lại RGB -> BGR để hiển thị
    
    Trả về:
    - image: ảnh đã xử lý
    - results: kết quả chứa toạ độ các điểm (pose, tay trái, tay phải, mặt)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# --- HÀM 2: TRÍCH XUẤT TOẠ ĐỘ THÀNH MẢNG SỐ ---
def extract_keypoints(results):
    """
    Nhận vào: results từ MediaPipe
    
    Xử lý:
    - Nếu phát hiện được cơ thể (pose_landmarks tồn tại):
      Lấy 33 điểm, mỗi điểm có (x, y, z, visibility)
      Chuyển thành mảng phẳng: [x0, y0, z0, v0, x1, y1, z1, v1, ...]
    
    - Nếu KHÔNG phát hiện được:
      Tạo mảng toàn số 0 (33*4 = 132 số)
    
    - Tương tự cho tay trái (21 điểm, mỗi điểm x,y,z = 63 số)
    - Tương tự cho tay phải (21 điểm = 63 số)
    
    Trả về: Mảng 1 chiều gồm 258 số (132 + 63 + 63)
    """
    # POSE (cơ thể): 33 điểm * 4 = 132 số
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
    
    # TAY TRÁI: 21 điểm * 3 = 63 số
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)
    
    # TAY PHẢI: 21 điểm * 3 = 63 số
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    
    # Ghép tất cả lại
    return np.concatenate([pose, lh, rh])


# --- CHƯƠNG TRÌNH CHÍNH ---
def main():
    print("=" * 60)
    print("CHƯƠNG TRÌNH TEST 1 VIDEO")
    print("=" * 60)
    
    # Mở video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"❌ Không mở được video: {VIDEO_PATH}")
        print("   Hãy đảm bảo file tồn tại!")
        return
    
    # Lấy thông tin video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"\n📹 Video: {VIDEO_PATH}")
    print(f"   Tổng số frame: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   Thời lượng: ~{total_frames/fps:.2f} giây")
    print("\n⏳ Đang xử lý...\n")
    
    frames_data = []  # List chứa dữ liệu của TẤT CẢ các frame
    
    # Khởi tạo MediaPipe
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,  # Độ tin cậy tối thiểu khi phát hiện
        min_tracking_confidence=0.5    # Độ tin cậy khi theo dõi
    ) as holistic:
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # BƯỚC 1: Phát hiện keypoints
            image, results = mediapipe_detection(frame, holistic)
            
            # BƯỚC 2: Trích xuất thành mảng số
            keypoints = extract_keypoints(results)
            frames_data.append(keypoints)
            
            # In tiến trình
            if frame_count % 10 == 0:
                print(f"   Frame {frame_count}/{total_frames}")
            
            # Vẽ skeleton lên ảnh (để xem)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Hiển thị
            cv2.imshow('Đang xử lý - Nhấn Q để thoát', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # BƯỚC 3: Chuyển list -> numpy array và lưu
    np_data = np.array(frames_data)
    np.save(OUTPUT_FILE, np_data)
    
    # --- PHÂN TÍCH CẤU TRÚC DỮ LIỆU ---
    print("\n" + "=" * 60)
    print("✅ HOÀN THÀNH!")
    print("=" * 60)
    print(f"\n📊 CẤU TRÚC DỮ LIỆU ĐÃ LƯU:")
    print(f"   File: {OUTPUT_FILE}")
    print(f"   Shape (kích thước): {np_data.shape}")
    print(f"   └─> ({np_data.shape[0]} frames, {np_data.shape[1]} keypoints/frame)")
    print(f"\n   Giải thích:")
    print(f"   - Có {np_data.shape[0]} frames được xử lý")
    print(f"   - Mỗi frame có {np_data.shape[1]} số")
    print(f"     + 132 số đầu: Pose (33 điểm x 4)")
    print(f"     + 63 số tiếp: Tay trái (21 điểm x 3)")
    print(f"     + 63 số cuối: Tay phải (21 điểm x 3)")
    
    print(f"\n📝 VÍ DỤ DỮ LIỆU FRAME ĐẦU TIÊN:")
    print(f"   Frame 0 có {len(np_data[0])} giá trị")
    print(f"   10 giá trị đầu: {np_data[0][:10]}")
    print(f"   (Đây là toạ độ x,y,z,visibility của 2.5 điểm đầu tiên ở pose)")
    
    print(f"\n💾 Để load lại dữ liệu:")
    print(f"   data = np.load('{OUTPUT_FILE}')")
    print(f"   print(data.shape)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
