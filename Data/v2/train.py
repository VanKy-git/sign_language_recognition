import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Bidirectional, 
                                     LSTM, Dense, Dropout, BatchNormalization, 
                                     MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from sklearn.metrics import classification_report, confusion_matrix
# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PATH, "npy_datav2")
MODEL_DIR = os.path.join(PATH, "model")
FRAMES = 50
FEATURES = 504  # 252 tọa độ/góc + 252 vận tốc

def load_and_prepare_data(data_path):
    actions = sorted(os.listdir(data_path))
    label_map = {label:num for num, label in enumerate(actions)}
    X, y = [], []
    for action in actions:
        action_path = os.path.join(data_path, action)
        if not os.path.isdir(action_path): continue
        for file_name in os.listdir(action_path):
            if file_name.endswith('.npy'):
                X.append(np.load(os.path.join(action_path, file_name)))
                y.append(label_map[action])
    return np.array(X), to_categorical(y, num_classes=len(actions)).astype(int), label_map

class SignLanguageDataGen(Sequence):
    def __init__(self, X, y, batch_size, mean, std, augment=False):
        self.X, self.y, self.batch_size = X, y, batch_size
        self.mean, self.std, self.augment = mean, std, augment
        self.indices = np.arange(len(self.X))
        np.random.shuffle(self.indices)

    def __len__(self): return int(np.floor(len(self.X) / self.batch_size))
    def on_epoch_end(self): np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        X_batch, y_batch = self.X[batch_indices].copy(), self.y[batch_indices]

        if self.augment:
            for i in range(len(X_batch)):
                scale_factor = np.random.uniform(0.85, 1.15)
                X_batch[i] = X_batch[i] * scale_factor
                X_batch[i] += np.random.normal(0, 0.005, X_batch[i].shape)

        return (X_batch - self.mean) / (self.std + 1e-7), y_batch

if __name__ == "__main__":
    X, y, label_map = load_and_prepare_data(DATA_PATH)
    num_classes = len(label_map)  
    print(f"Tổng video: {X.shape[0]}, Lớp: {num_classes}, Features/Frame: {FEATURES}")

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

    train_mean, train_std = np.mean(X_train, axis=(0, 1)), np.std(X_train, axis=(0, 1))
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(os.path.join(MODEL_DIR, 'train_mean.npy'), train_mean)
    np.save(os.path.join(MODEL_DIR, 'train_std.npy'), train_std)

    train_gen = SignLanguageDataGen(X_train, y_train, 32, train_mean, train_std, augment=True)
    val_gen = SignLanguageDataGen(X_val, y_val, 32, train_mean, train_std, augment=False)

    # ======================================================================
    # KIẾN TRÚC MỚI: 1D-CNN + Bi-LSTM + TRANSFORMER (Self-Attention)
    # ======================================================================
    inputs = Input(shape=(FRAMES, FEATURES))
    
# 1. Khối CNN (Giảm filters xuống 64 -> 128)
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # 2. Khối LSTM (Bỏ bớt 1 lớp LSTM 128, chỉ xài 1 lớp 64)
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)))(x)
    x = Dropout(0.4)(x)

    # 3. Khối TRANSFORMER (Giảm Head xuống 2, Key_dim 32)
    attn_out = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = LayerNormalization()(x + attn_out) 
    x = GlobalAveragePooling1D()(x)

    # 4. Khối Output
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Khởi tạo mô hình
    model = Model(inputs=inputs, outputs=outputs)
    # ======================================================================

    model.compile(
        optimizer='adam', 
        loss=CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25), 
        metrics=['categorical_accuracy']
    )    
    # In ra cấu trúc mạng để ông thấy sự xuất hiện của lớp MultiHeadAttention
    model.summary()

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, f'hybrid_model_{num_classes}tu.keras'), save_best_only=True)

    model.fit(train_gen, validation_data=val_gen, epochs=200, callbacks=[early_stop, checkpoint, lr_scheduler])

    X_test_norm = (X_test - train_mean) / (train_std + 1e-7)
    test_loss, test_acc = model.evaluate(X_test_norm, y_test, batch_size=32)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("\n======================================================")
    print("6. BÁO CÁO CHI TIẾT LỖI SAI TỪNG TỪ (CLASSIFICATION REPORT)")
    print("======================================================")
    
    print("Đang chạy dự đoán trên tập Test để phân tích...")
    # Dự đoán thẳng trên mảng X_test_norm
    y_pred_probs = model.predict(X_test_norm, batch_size=32, verbose=0)
    
    # Lấy index của class có xác suất cao nhất
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Lấy lại danh sách từ vựng theo đúng thứ tự ID
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    
    # In báo cáo F1-Score, Recall, Precision
    report = classification_report(y_true_classes, y_pred_classes, target_names=target_names, zero_division=0)
    print(report)
    
    print("\n--- TÌM CẶP TỪ BỊ NHẦM LẪN NHIỀU NHẤT ---")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    confused_pairs = []
    
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((target_names[i], target_names[j], cm[i, j]))
                
    # Sắp xếp theo số lần nhầm lẫn giảm dần để ưu tiên fix trước
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if len(confused_pairs) == 0:
        print("Tuyệt vời! Model không nhầm lẫn cặp từ nào trên tập Test.")
    else:
        for pair in confused_pairs:
            print(f"-> Dấu '{pair[0]}' bị AI đoán nhầm thành '{pair[1]}' ({pair[2]} lần)")
            
    print("\n=== HOÀN TẤT HUẤN LUYỆN VÀ ĐÁNH GIÁ ===")