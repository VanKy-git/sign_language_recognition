from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import numpy as np

X = np.load('X_10tu.npy')
y = np.load('y_10tu_optimized.npy')
model = Sequential()

model.add(LSTM(64,activation = 'tanh', input_shape=(60, 258), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='tanh', return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

#Compile model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
# monitor='val_loss': Theo dõi điểm số của bài thi thử (Validation Loss).
# patience=15: Nếu qua 15 vòng liên tiếp mà điểm thi thử không bứt phá lên được, lập tức DỪNG HỌC.
# restore_best_weights=True: Trả lại cho em bộ não ở cái vòng mà nó thông minh nhất, vứt bỏ những vòng ngu ngốc phía sau.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Bắt đầu huấn luyện...")
history = model.fit(X_train, y_train, 
                    epochs=200, 
                    batch_size=32, 
                    validation_data=(X_test, y_test), 
                    callbacks=[early_stop])

# X_train, y_train: Sách giáo khoa và Sách giải để nó học.
# epochs=200: Cho phép nó đọc đi đọc lại cuốn sách giáo khoa này tối đa 200 lần.
# batch_size=32: Không bắt nó nạp cả 600 video cùng lúc (cháy RAM mất). Bắt nó lấy 32 video ra học -> vặn núm sửa lỗi 1 lần -> lấy 32 video tiếp theo...
# validation_data=(X_test, y_test): Đây là Đề thi cuối kỳ. Sau mỗi 1 vòng (Epoch), nó sẽ bị lôi ra làm bài thi này. Nếu nó học vẹt, điểm thi tập Train sẽ cao nhưng điểm thi tập Test này sẽ cực thấp.
# callbacks=[early_stop]: Gắn cái phanh ta vừa tạo ở trên vào quá trình chạy.

model.save('model/model4.keras')