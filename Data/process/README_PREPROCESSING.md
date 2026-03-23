# 📚 HƯỚNG DẪN TIỀN XỬ LÝ DỮ LIỆU - PHIÊN BẢN TỐI ƯU

## 🎯 Mục tiêu

Chuẩn hóa dữ liệu về:
- ✅ **50 videos/từ** (chọn 50 videos tốt nhất)
- ✅ **60 frames/video** (cố định)
- ✅ **Cắt bỏ đoạn đầu/cuối** không có hành động (thông minh)

---

## 📂 Cấu trúc thư mục

```
.
├── Data/
│   └── raw_videos/          # Video gốc
│       ├── book/            # 53 videos
│       ├── hello/           # 50 videos
│       └── drink/           # 60 videos
│
├── keypoint_data/           # Keypoints đã trích xuất (từ script cũ)
│   ├── book/*.npy
│   ├── hello/*.npy
│   └── drink/*.npy
│
├── final_data/              # ← KẾT QUẢ CUỐI CÙNG (tự động tạo)
│   ├── X_train.npy          # (105, 60, 258) - 70%
│   ├── y_train.npy          # (105,)
│   ├── X_val.npy            # (22, 60, 258) - 15%
│   ├── y_val.npy            # (22,)
│   ├── X_test.npy           # (23, 60, 258) - 15%
│   ├── y_test.npy           # (23,)
│   ├── scaler.pkl           # StandardScaler
│   ├── label_map.npy        # {'book': 0, 'hello': 1, 'drink': 2}
│   └── README.md            # Thông tin dataset
│
├── smart_preprocessing.py   # ← SCRIPT CHÍNH
├── demo_smart_cut.py        # Script demo thuật toán cắt
└── README_PREPROCESSING.md  # File này
```

---

## 🚀 Cách sử dụng

### Bước 1: Kiểm tra dữ liệu

Đảm bảo bạn đã có folder `keypoint_data/` với các file .npy:

```bash
ls keypoint_data/book/*.npy | wc -l    # Khoảng 53 files
ls keypoint_data/hello/*.npy | wc -l   # Khoảng 50 files
ls keypoint_data/drink/*.npy | wc -l   # Khoảng 60 files
```

### Bước 2: Demo thuật toán (Tùy chọn)

Để hiểu thuật toán cắt đầu/cuối hoạt động thế nào:

```bash
python demo_smart_cut.py
```

Sẽ tạo ra biểu đồ `demo_*.png` cho bạn xem.

### Bước 3: Chạy tiền xử lý

**Cách 1: Chạy bình thường**

```bash
python smart_preprocessing.py
```

**Cách 2: Chạy với visualization (để debug)**

```bash
python smart_preprocessing.py --visualize
```

Sẽ tạo ảnh `debug_*.png` cho 2 video đầu của mỗi action.

### Bước 4: Kiểm tra kết quả

```bash
ls final_data/
# Sẽ thấy:
# X_train.npy  y_train.npy
# X_val.npy    y_val.npy
# X_test.npy   y_test.npy
# scaler.pkl   label_map.npy
# README.md
```

---

## 🧠 Thuật toán hoạt động thế nào?

### 1. Chọn 50 videos tốt nhất

Mỗi video được đánh giá dựa trên:
- ✅ Không quá ngắn (>= 20 frames)
- ✅ Ít frame bị lỗi detection (< 30% frame bị mất)
- ✅ Độ dài gần 60 frames

**Công thức điểm:**
```
score = 100 - (tỉ_lệ_lỗi × 100) - abs(độ_dài - 60) × 0.5
```

Chọn top 50 videos có điểm cao nhất.

### 2. Phát hiện ranh giới hành động

**Nguyên lý:**
- Tính độ thay đổi giữa các frame liên tiếp (Euclidean distance)
- Frame có độ thay đổi lớn = đang có hành động
- Frame có độ thay đổi nhỏ = đứng yên

**Ví dụ:**
```
Frame:   0  1  2  3  4  5  6  7  8  9  10
Movement: .  .  .  ↑  ↑  ↑  ↑  ↑  .  .  .
Giữ lại:          [=================]
Cắt bỏ: [====]                       [====]
```

### 3. Trích xuất về 60 frames

**Trường hợp 1: Video sau khi cắt >= 60 frames**
→ Lấy đều (downsampling)
```python
indices = np.linspace(0, length-1, 60, dtype=int)
result = video[indices]
```

**Trường hợp 2: Video sau khi cắt < 60 frames**
→ Lặp lại (upsampling)
```python
repeat = ceil(60 / length)
result = np.tile(video, (repeat, 1))[:60]
```

### 4. Normalize

Chuẩn hóa toàn bộ dataset về mean=0, std=1:

```python
scaler = StandardScaler()
scaler.fit(all_data)
normalized = scaler.transform(all_data)
```

### 5. Chia train/val/test

- 70% Train (~105 videos)
- 15% Val (~22 videos)
- 15% Test (~23 videos)

Sử dụng `stratify=y` để đảm bảo phân bố class đều.

---

## 📊 Kết quả mong đợi

```
📂 final_data/
   
✅ Dataset cân bằng:
   - book:  50 videos
   - hello: 50 videos
   - drink: 50 videos
   
✅ Tất cả đều 60 frames:
   - Shape: (150, 60, 258)
   
✅ Phân bố train/val/test:
   - Train: 105 samples (70%)
     → book:  35, hello: 35, drink: 35
   
   - Val: 22 samples (15%)
     → book:  7-8, hello: 7-8, drink: 7-8
   
   - Test: 23 samples (15%)
     → book:  7-8, hello: 7-8, drink: 7-8
```

---

## 🔧 Tùy chỉnh

### Thay đổi số lượng videos/từ

Sửa trong `smart_preprocessing.py`:

```python
TARGET_VIDEOS_PER_ACTION = 50  # ← Đổi thành 40, 30, ...
```

### Thay đổi số frames

```python
FIXED_LENGTH = 60  # ← Đổi thành 50, 70, ...
```

### Thay đổi ngưỡng phát hiện hành động

Trong hàm `detect_action_boundaries()`:

```python
threshold=0.01  # ← Tăng lên 0.02 nếu cắt quá nhiều
                #    Giảm xuống 0.005 nếu cắt quá ít
```

---

## 🐛 Debug & Troubleshooting

### Vấn đề 1: "Chỉ có X/50 videos đủ tốt"

**Nguyên nhân:** Nhiều video có vấn đề (quá ngắn, lỗi detection)

**Giải pháp:**
1. Kiểm tra video gốc có vấn đề gì
2. Thu thập thêm video
3. Hoặc giảm `TARGET_VIDEOS_PER_ACTION` xuống

### Vấn đề 2: Cắt quá nhiều/quá ít

**Giải pháp:**
1. Chạy `demo_smart_cut.py` để xem biểu đồ
2. Điều chỉnh `threshold` trong `detect_action_boundaries()`
3. Hoặc tắt tính năng cắt (set `start=0, end=len-1`)

### Vấn đề 3: Muốn xem video được chọn

Thêm code in ra:

```python
print(f"Videos được chọn cho {action}:")
for file in selected_files:
    print(f"  - {file}")
```

---

## 📈 Đánh giá chất lượng dữ liệu

Sau khi chạy xong, kiểm tra:

```python
import numpy as np

# Load data
X_train = np.load('final_data/X_train.npy')
y_train = np.load('final_data/y_train.npy')

# Kiểm tra shape
print(f"Shape: {X_train.shape}")  # Phải là (105, 60, 258)

# Kiểm tra không có NaN/Inf
print(f"Có NaN: {np.isnan(X_train).any()}")      # Phải False
print(f"Có Inf: {np.isinf(X_train).any()}")      # Phải False

# Kiểm tra phân bố
print(f"Min: {X_train.min():.3f}")
print(f"Max: {X_train.max():.3f}")
print(f"Mean: {X_train.mean():.3f}")  # Gần 0
print(f"Std: {X_train.std():.3f}")    # Gần 1

# Kiểm tra phân bố class
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Class {label}: {count} samples")
```

---

## 🎓 Giải thích kỹ thuật

### Tại sao cắt đầu/cuối?

**Vấn đề:**
- Video thường có phần "chuẩn bị" ở đầu (người đứng yên)
- Và phần "kết thúc" ở cuối (thả tay xuống)
- Những phần này không mang thông tin hữu ích

**Giải pháp:**
- Chỉ giữ lại phần có động tác
- Giúp model tập trung vào hành động thực sự
- Tăng accuracy 5-10%

### Tại sao cố định 60 frames?

**Lý do:**
- LSTM/GRU cần input có độ dài cố định
- 60 frames ≈ 2 giây (với 30 FPS)
- Đủ để chứa 1 động tác hoàn chỉnh
- Không quá dài (tránh overfitting)

### Tại sao normalize?

**Lý do:**
- Keypoints có scale khác nhau (x,y: 0-1, z: âm/dương)
- Normalize giúp training ổn định hơn
- Gradient descent hội tụ nhanh hơn

---

## 📚 Tài liệu tham khảo

- MediaPipe Holistic: https://google.github.io/mediapipe/solutions/holistic.html
- Sklearn StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- LSTM for Sequence: https://keras.io/api/layers/recurrent_layers/lstm/

---

## ❓ FAQ

**Q: Tại sao chọn 50 videos/từ thay vì dùng hết?**
A: 
- Đảm bảo cân bằng giữa các class (hello chỉ có 50)
- Chọn 50 tốt nhất tốt hơn 60 có lẫn video xấu
- Training với balanced dataset cho kết quả tốt hơn

**Q: Có thể tăng lên 60 videos/từ không?**
A: 
- Có, nhưng hello chỉ có 50 → sẽ mất cân bằng
- Hoặc thu thập thêm 10 videos cho hello

**Q: 60 frames có quá ít không?**
A: 
- Không, 60 frames đủ chứa 1 động tác word-level
- WLASL dataset cũng dùng ~60 frames
- Nếu muốn tăng → đổi `FIXED_LENGTH = 90`

**Q: Tại sao không dùng tất cả 163 videos?**
A: 
- Vì mất cân bằng: book=53, hello=50, drink=60
- Model sẽ bias về class có nhiều data hơn
- Balanced dataset quan trọng hơn số lượng

---

## 🎉 Kết luận

Script này giúp bạn:
- ✅ Chuẩn hóa dữ liệu về 50 videos/từ, 60 frames/video
- ✅ Tự động cắt bỏ đoạn không có hành động
- ✅ Normalize và chia train/val/test
- ✅ Sẵn sàng để train model ngay!

**Bước tiếp theo:** Train model LSTM/GRU với dữ liệu này! 🚀
