import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalFocalCrossentropy

# --- CẤU HÌNH ---
DATA_PATH = "npy_datas"
CLASSES = 30           # Tạm thời huấn luyện 10 từ trước
FRAMES = 50
FEATURES = 384         # 162 tọa độ + 162 vận tốc

# --- 1. HÀM GOM DỮ LIỆU ---
def load_and_prepare_data(data_path):
    actions = sorted(os.listdir(data_path))
    label_map = {label:num for num, label in enumerate(actions)}
    
    X, y = [], []
    for action in actions:
        action_path = os.path.join(data_path, action)
        if not os.path.isdir(action_path): continue
        for file_name in os.listdir(action_path):
            if file_name.endswith('.npy'):
                res = np.load(os.path.join(action_path, file_name))
                X.append(res)
                y.append(label_map[action])
                
    X = np.array(X)
    y = to_categorical(y, num_classes=len(actions)).astype(int)
    return X, y, label_map

# --- 2. CUSTOM DATA GENERATOR (Augmentation & Normalization) ---
class SignLanguageDataGen(Sequence):
    def __init__(self, X, y, batch_size, mean, std, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.indices = np.arange(len(self.X))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        # Lấy batch hiện tại
        batch_indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices]

        # Áp dụng Augmentation (chỉ cho tập Train)
        if self.augment:
            for i in range(len(X_batch)):
                # 1. Random Scaling (Giả lập người to/nhỏ): Phóng to/thu nhỏ khung xương từ 0.8x đến 1.2x
                scale_factor = np.random.uniform(0.8, 1.2)
                X_batch[i] = X_batch[i] * scale_factor
                
                # 2. Random Jittering (Nhiễu không gian): Thêm nhiễu Gaussian nhẹ vào tọa độ
                noise = np.random.normal(loc=0.0, scale=0.01, size=X_batch[i].shape)
                X_batch[i] = X_batch[i] + noise

        # Áp dụng Z-score Normalization (Dùng Mean/Std của tập Train)
        # Cộng epsilon (1e-7) vào std để tránh chia cho 0
        X_batch = (X_batch - self.mean) / (self.std + 1e-7)

        return X_batch, y_batch

# --- 3. QUY TRÌNH CHÍNH ---
if __name__ == "__main__":
    print("1. Đang gom dữ liệu từ folder...")
    X, y, label_map = load_and_prepare_data(DATA_PATH)
    print(f"Tổng số video: {X.shape[0]}, Shape: {X.shape}")
    print(f"Map nhãn: {label_map}")

    # Chia tập: Train (70%), Val (15%), Test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42) # ~15% tổng

    print("2. Đang tính toán cấu hình chuẩn hóa (Mean, Std) từ tập Train...")
    # Tính Mean và Std trên toàn bộ các frames của tập Train
    train_mean = np.mean(X_train, axis=(0, 1)) 
    train_std = np.std(X_train, axis=(0, 1))

    # LƯU CẤU HÌNH ĐỂ CHẠY REAL-TIME
    os.makedirs('model', exist_ok=True)
    np.save('model/train_mean.npy', train_mean)
    np.save('model/train_std.npy', train_std)
    print("Đã lưu 'train_mean.npy' và 'train_std.npy' vào thư mục model/.")

    # Khởi tạo Generators
    train_gen = SignLanguageDataGen(X_train, y_train, batch_size=32, mean=train_mean, std=train_std, augment=True)
    val_gen = SignLanguageDataGen(X_val, y_val, batch_size=32, mean=train_mean, std=train_std, augment=False)
    test_gen = SignLanguageDataGen(X_test, y_test, batch_size=32, mean=train_mean, std=train_std, augment=False)

    print("3. Khởi tạo mô hình Hybrid (1D-CNN + Bi-LSTM)...")
    model = Sequential([
        # Tăng filter lên 128 để đọc được nhiều đặc trưng hơn (từ 384 dimensions)
        Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(FRAMES, FEATURES)),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.4), # Tăng chút dropout chống overfitting

        # Giữ nguyên phần Bi-LSTM
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))),
        Dropout(0.4),
        
        Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01))),
        Dropout(0.4),

        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(CLASSES, activation='softmax')
    ])

    # ĐỔI LẠI THÀNH CATEGORICAL CROSSENTROPY
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    checkpoint = ModelCheckpoint('model/hybrid_model_30tu.keras', monitor='val_loss', save_best_only=True)

    print("4. Bắt đầu huấn luyện...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=150,
        callbacks=[early_stop, checkpoint]
    )

    print("\n5. Đánh giá trên tập Test (Dữ liệu chưa từng thấy)...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n======================================================")
    print("6. BÁO CÁO CHI TIẾT LỖI SAI TỪNG TỪ (CLASSIFICATION REPORT)")
    print("======================================================")
    
    # Gom dữ liệu thực tế và dự đoán từ tập Test
    y_true = []
    y_pred = []
    
    print("Đang chạy dự đoán trên tập Test để phân tích...")
    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]
        preds = model.predict(X_batch, verbose=0)
        
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Lấy lại danh sách từ vựng theo đúng thứ tự (0, 1, 2...)
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    
    # In báo cáo
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print(report)
    
    print("\n--- TÌM CẶP TỪ BỊ NHẦM LẪN NHIỀU NHẤT ---")
    cm = confusion_matrix(y_true, y_pred)
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            if i != j and cm[i, j] > 0:
                print(f"-> Chữ '{target_names[i]}' bị AI đoán nhầm thành '{target_names[j]}' ({cm[i, j]} lần)")