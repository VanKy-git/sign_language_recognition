import os
import json
import numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import google.generativeai as genai

# Kế thừa lõi xử lý đặc trưng
from feature_engineering import process_single_video_features

# ============= CẤU HÌNH HỆ THỐNG =============
ACTIONS = [
    'Eat', 'Father - Dad', 'Mother - Mom', 'Student', 'ThankYou', 'baby', 'big', 
    'book', 'bread', 'brother', 'call', 'cold', 'difficult', 'drink', 'easy', 
    'fast', 'give', 'good', 'goodbye', 'happy', 'hello', 'home-house', 'hot', 
    'hunggry', 'listen', 'love', 'me', 'open', 'play', 'read', 'rice', 'run', 
    'sad', 'school', 'sister', 'sit', 'sleep', 'slow', 'small', 'sorry', 
    'teacher', 'tired', 'toilet-WC', 'wash', 'we', 'what', 'when', 'where', 
    'why', 'you'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'hybrid_model_50tu.keras')
MEAN_PATH = os.path.join(BASE_DIR, 'model', 'train_mean.npy')
STD_PATH = os.path.join(BASE_DIR, 'model', 'train_std.npy')

START_THRESHOLD = 0.04
STOP_THRESHOLD = 0.02
MIN_ACTION_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.75

# ============= CẤU HÌNH GEMINI =============
# ⚠️ CẢNH BÁO: HÃY ĐỔI KEY NÀY SAU KHI TEST XONG VÌ NÓ ĐÃ BỊ LỘ TRÊN MẠNG
GEMINI_API_KEY = "AIzaSyCfyEDOG5UqoeSyXIgnZGWzzjlU2cUFKUU"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")
# ✅ SỬA LỖI 1: Thêm chữ 'async' và đổi thành hàm 'generate_content_async' 
# để cho phép Gemini chạy ngầm, không làm giật lag Server và đứt kết nối Camera.
async def generate_natural_sentence(keywords):
    if not keywords: return ""
    prompt = f"Ghép các từ khóa ngôn ngữ ký hiệu sau thành một câu giao tiếp tiếng Việt tự nhiên, ngắn gọn và có nghĩa. Chỉ trả về câu kết quả, tuyệt đối không giải thích thêm. Từ khóa: {', '.join(keywords)}"
    try:
        response = await gemini_model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Lỗi gọi Gemini: {e}")
        return " ".join(keywords)

# ============= KHỞI TẠO FASTAPI =============
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

print("Đang tải Model và cấu hình chuẩn hóa Z-Score...")
model = load_model(MODEL_PATH)
train_mean = np.load(MEAN_PATH)
train_std = np.load(STD_PATH)
print("Sẵn sàng nhận kết nối từ Frontend!")

def calculate_motion_score(buffer):
    if len(buffer) < 2: return 0.0
    seq = np.array(buffer)
    hands_only = seq[:, 36:] 
    diffs = np.abs(np.diff(hands_only, axis=0))
    return np.mean(np.sum(diffs, axis=1))

@app.websocket("/ws/sign-language")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    motion_buffer = deque(maxlen=5)
    action_frames = []
    is_recording = False
    sentence = []
    
    try:
        while True:
            data = await websocket.receive_text()
            raw_keypoints = np.array(json.loads(data))
            
            motion_buffer.append(raw_keypoints)
            current_motion = calculate_motion_score(list(motion_buffer)) if len(motion_buffer) == 5 else 0
            
            if not is_recording:
                if current_motion > START_THRESHOLD:
                    is_recording = True
                    action_frames = list(motion_buffer)
                    await websocket.send_json({"event": "recording_start"})
            else:
                action_frames.append(raw_keypoints)
                if current_motion < STOP_THRESHOLD or len(action_frames) > 100:
                    is_recording = False
                    await websocket.send_json({"event": "recording_stop"})
                    
                    if len(action_frames) >= MIN_ACTION_FRAMES:
                        features = process_single_video_features(action_frames)
                        
                        if features is not None:
                            features_norm = (features - train_mean) / (train_std + 1e-7)
                            X_input = np.expand_dims(features_norm, axis=0)
                            
                            preds = model.predict(X_input, verbose=0)[0]
                            best_idx = np.argmax(preds)
                            best_conf = preds[best_idx]
                            
                            if best_conf > CONFIDENCE_THRESHOLD:
                                current_action = ACTIONS[best_idx]
                                
                                # Cập nhật mảng từ khóa: Chỉ thêm nếu từ mới khác từ cũ
                                if not sentence or sentence[-1] != current_action:
                                    sentence.append(current_action)
                                    if len(sentence) > 5:
                                        sentence = sentence[-5:]
                                        
                                # ✅ SỬA LỖI 2: Đưa phần gọi Gemini và Gửi JSON ra khỏi vòng lặp if.
                                # Nhờ vậy, dù ông múa 1 từ trùng lặp, server vẫn báo kết quả về cho Web thay vì im lìm.
                                natural_sentence = await generate_natural_sentence(sentence)
                                
                                await websocket.send_json({
                                    "event": "prediction_success",
                                    "action": current_action,
                                    "confidence": float(best_conf),
                                    "keywords": sentence,
                                    "natural_sentence": natural_sentence
                                })
                            else:
                                await websocket.send_json({"event": "low_confidence"})
                    else:
                        await websocket.send_json({"event": "action_too_short"})
                        
                    action_frames = [] # Reset vòng lặp
                    
    except WebSocketDisconnect:
        print("Frontend đã ngắt kết nối WebSocket.")