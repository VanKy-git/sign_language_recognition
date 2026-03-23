# GIẢI THÍCH CHI TIẾT CẤU TRÚC DỮ LIỆU

## 📌 Tổng quan

File `.npy` lưu dữ liệu dạng mảng 2 chiều:
```
Shape: (số_frames, 258)
```

**Ví dụ:** Video 30 frame → Shape = (30, 258)

---

## 🔍 Mỗi frame có 258 số được chia thành 3 phần:

### 1️⃣ POSE (Cơ thể) - 132 số đầu tiên (index 0-131)

MediaPipe phát hiện **33 điểm** trên cơ thể:
- Mặt: mũi, mắt, tai, miệng
- Vai, khuỷu tay, cổ tay
- Hông, đầu gối, mắt cá chân
- ...

**Mỗi điểm có 4 giá trị:**
- `x`: Toạ độ ngang (0.0 - 1.0, chuẩn hoá theo chiều rộng ảnh)
- `y`: Toạ độ dọc (0.0 - 1.0, chuẩn hoá theo chiều cao ảnh)
- `z`: Độ sâu (khoảng cách từ camera, có thể âm/dương)
- `visibility`: Độ "nhìn thấy được" (0.0 - 1.0)
  - 1.0 = rõ ràng
  - 0.0 = bị che khuất hoặc ngoài khung hình

**Công thức:**
```
33 điểm × 4 giá trị = 132 số
```

**Cấu trúc trong mảng:**
```python
pose[0:4]   = [x0, y0, z0, visibility0]  # Điểm 0 (mũi)
pose[4:8]   = [x1, y1, z1, visibility1]  # Điểm 1 (mắt trái trong)
pose[8:12]  = [x2, y2, z2, visibility2]  # Điểm 2 (mắt trái)
...
pose[128:132] = [x32, y32, z32, visibility32]  # Điểm 32 (mắt cá chân phải)
```

---

### 2️⃣ LEFT HAND (Tay trái) - 63 số tiếp theo (index 132-194)

MediaPipe phát hiện **21 điểm** trên bàn tay trái:
- Cổ tay (1 điểm)
- Ngón cái (4 điểm)
- Ngón trỏ (4 điểm)
- Ngón giữa (4 điểm)
- Ngón áp út (4 điểm)
- Ngón út (4 điểm)

**Mỗi điểm có 3 giá trị:**
- `x`: Toạ độ ngang (0.0 - 1.0)
- `y`: Toạ độ dọc (0.0 - 1.0)
- `z`: Độ sâu

**Công thức:**
```
21 điểm × 3 giá trị = 63 số
```

**Lưu ý:** Nếu KHÔNG phát hiện được tay trái → 63 số đều = 0

---

### 3️⃣ RIGHT HAND (Tay phải) - 63 số cuối cùng (index 195-257)

Tương tự tay trái, cũng **21 điểm × 3 giá trị = 63 số**

---

## 💡 Ví dụ cụ thể

Giả sử video có 5 frame:

```python
data = np.load('test_output.npy')
print(data.shape)  # Output: (5, 258)

# Frame đầu tiên
frame_0 = data[0]  # Mảng 258 số

# Tách từng phần
pose_data = frame_0[0:132]      # 132 số pose
left_hand = frame_0[132:195]    # 63 số tay trái
right_hand = frame_0[195:258]   # 63 số tay phải

# Xem toạ độ điểm mũi (điểm 0 của pose)
nose_x = frame_0[0]
nose_y = frame_0[1]
nose_z = frame_0[2]
nose_visibility = frame_0[3]

print(f"Mũi ở vị trí: x={nose_x:.3f}, y={nose_y:.3f}, z={nose_z:.3f}")
```

---

## 🎯 Tại sao lưu dạng này?

### ✅ Ưu điểm:
1. **Chuẩn hoá**: Toạ độ 0-1 → không phụ thuộc kích thước video
2. **Compact**: Chỉ lưu số, không lưu ảnh → tiết kiệm dung lượng
3. **Dễ train AI**: Mảng số đều → dễ đưa vào mạng neural (LSTM, GRU...)
4. **Xử lý nhanh**: NumPy tính toán cực nhanh trên mảng

### ⚠️ Lưu ý:
- Nếu video 100 frame → mảng (100, 258) → 25,800 số
- File .npy sẽ nặng khoảng 200-500 KB/video (nhẹ hơn video gốc rất nhiều!)

---

## 🧪 Cách kiểm tra dữ liệu

```python
import numpy as np

# Load file
data = np.load('test_output.npy')

# Xem thông tin cơ bản
print("Shape:", data.shape)
print("Kiểu dữ liệu:", data.dtype)
print("Min value:", data.min())
print("Max value:", data.max())

# Kiểm tra có frame nào toàn 0 không (= không phát hiện được gì)
zero_frames = np.where(data.sum(axis=1) == 0)[0]
print(f"Số frame không phát hiện được: {len(zero_frames)}")

# Trực quan hóa frame thứ 10
import matplotlib.pyplot as plt
plt.plot(data[10])
plt.title("258 giá trị của frame 10")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()
```

---

## 📚 Tài liệu tham khảo

- MediaPipe Pose Landmarks: https://google.github.io/mediapipe/solutions/pose.html
- MediaPipe Hand Landmarks: https://google.github.io/mediapipe/solutions/hands.html

---

## ❓ FAQ

**Q: Tại sao visibility quan trọng?**
A: Nếu visibility thấp (< 0.3), có thể điểm đó bị che khuất → không nên tin tưởng toạ độ đó khi train model.

**Q: Tại sao pose có 4 giá trị mà hand chỉ có 3?**
A: MediaPipe Pose trả về visibility, còn Hand không (giả định tay luôn nhìn thấy rõ nếu phát hiện được).

**Q: z âm/dương có nghĩa gì?**
A: 
- z > 0: Điểm đó xa camera hơn điểm reference (thường là hông)
- z < 0: Điểm đó gần camera hơn

**Q: Nếu không phát hiện được tay thì sao?**
A: Mảng 63 số tay đó sẽ toàn = 0.
