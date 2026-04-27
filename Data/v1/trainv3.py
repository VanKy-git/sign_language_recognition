import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "npy_datas")
MODEL_DIR = os.path.join(BASE_DIR, "model")
FRAMES = 50
FEATURES = 384         

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
        batch_indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices]

        if self.augment:
            for i in range(len(X_batch)):
                # 1. Scale chuẩn toán học (Scale cả tọa độ lẫn vận tốc)
                scale_factor = np.random.uniform(0.85, 1.15)
                X_batch[i] = X_batch[i] * scale_factor
                
                # 2. Nhiễu nhẹ (Jittering)
                X_batch[i] += np.random.normal(0, 0.005, X_batch[i].shape)

        # Z-score Normalization
        X_batch = (X_batch - self.mean) / (self.std + 1e-7)
        return X_batch, y_batch

if __name__ == "__main__":
    X, y, label_map = load_and_prepare_data(DATA_PATH)
    num_classes = len(label_map)  
    print(f"Tổng số video: {X.shape[0]}, Số lượng lớp: {num_classes}")

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

    train_mean = np.mean(X_train, axis=(0, 1)) 
    train_std = np.std(X_train, axis=(0, 1))
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(os.path.join(MODEL_DIR, 'train_mean.npy'), train_mean)
    np.save(os.path.join(MODEL_DIR, 'train_std.npy'), train_std)

    train_gen = SignLanguageDataGen(X_train, y_train, batch_size=32, mean=train_mean, std=train_std, augment=True)
    val_gen = SignLanguageDataGen(X_val, y_val, batch_size=32, mean=train_mean, std=train_std, augment=False)

    # TRẢ LẠI CẤU TRÚC LSTM CHUẨN
    model = Sequential([
        Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(FRAMES, FEATURES)),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.4),

        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))),
        Dropout(0.4),
        
        # Đã sửa lại return_sequences=False và XÓA GlobalAveragePooling
        Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01))),
        Dropout(0.4),

        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Trả lại Categorical Crossentropy cơ bản (Xóa Focal Loss và Class Weights)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, f'hybrid_model_{num_classes}tu.keras'), monitor='val_loss', save_best_only=True)

    history = model.fit(train_gen, validation_data=val_gen, epochs=200, callbacks=[early_stop, checkpoint, lr_scheduler])

    X_test_norm = (X_test - train_mean) / (train_std + 1e-7)
    test_loss, test_acc = model.evaluate(X_test_norm, y_test, batch_size=32)
    print(f"Test Accuracy: {test_acc*100:.2f}%")