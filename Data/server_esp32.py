import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import os
import json
import asyncio
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tensorflow.keras.models import load_model
import google.generativeai as genai

# Import lõi đặc trưng đã viết
from feature_engineering import process_single_video_features

# ============= CẤU HÌNH GEMINI =============
GEMINI_API_KEY = "DÁN_API_KEY_CỦA_ÔNG_VÀO_ĐÂY"  # <--- NHỚ ĐỔI LẠI KEY Ở ĐÂY
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')

# ============= CẤU HÌNH HỆ THỐNG =============
def load_actions_from_folders():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [os.path.join(base_dir, 'npy_datas'), os.path.join(base_dir, 'old_npydata'), os.path.join(base_dir, 'raw_videos')]
    data_dir = next((d for d in candidate_dirs if os.path.isdir(d)), None)
    if data_dir is None: return [] # Fallback
    return sorted(name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith('.'))

ACTIONS = load_actions_from_folders()
if not ACTIONS:
    ACTIONS = ['Eat', 'Father - Dad', 'Mother - Mom', 'Student', 'ThankYou', 'baby', 'big', 'book', 'bread', 'brother', 'call', 'cold', 'difficult', 'drink', 'easy', 'fast', 'give', 'good', 'goodbye', 'happy', 'hello', 'home-house', 'hot', 'hunggry', 'listen', 'love', 'me', 'open', 'play', 'read', 'rice', 'run', 'sad', 'school', 'sister', 'sit', 'sleep', 'slow', 'small', 'sorry', 'teacher', 'tired', 'toilet-WC', 'wash', 'we', 'what', 'when', 'where', 'why', 'you']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'hybrid_model_50tu.keras')
MEAN_PATH = os.path.join(BASE_DIR, 'model', 'train_mean.npy')
STD_PATH = os.path.join(BASE_DIR, 'model', 'train_std.npy')

ESP32_URL = "http://192.168.1.47:81/stream"  # <--- LINK ESP32 CỦA ÔNG
# ESP32_URL = 0 # Mở dòng này nếu muốn test bằng Webcam máy tính trước

# ============= FASTAPI & WEBSOCKET =============
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

connected_clients = set()
main_loop = None # Dùng để gọi hàm async từ thread OpenCV

@app.websocket("/ws/sign-language")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text() # Hứng ping từ client để giữ kết nối
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def broadcast_ws(message: dict):
    """Gửi dữ liệu JSON tới tất cả tab Web đang mở"""
    for client in list(connected_clients):
        try:
            await client.send_json(message)
        except:
            connected_clients.remove(client)

async def translate_and_broadcast(keywords):
    """Hàm gọi Gemini và đẩy thẳng câu dịch sang Web"""
    if not keywords: return
    print(f"\n[GEMINI] Đang dịch: {keywords} ...")
    prompt = f"Ghép các từ khóa ngôn ngữ ký hiệu sau thành một câu giao tiếp tiếng Việt tự nhiên, ngắn gọn và có nghĩa. Chỉ trả về câu kết quả, tuyệt đối không giải thích thêm. Từ khóa: {', '.join(keywords)}"
    try:
        response = await gemini_model.generate_content_async(prompt)
        natural_sentence = response.text.strip()
        print(f"🤖 KẾT QUẢ DỊCH: {natural_sentence}")
        # Bắn kết quả lên Web
        await broadcast_ws({
            "event": "translation_success",
            "keywords": keywords,
            "natural_sentence": natural_sentence
        })
    except Exception as e:
        print(f"[LỖI GEMINI]: {e}")

# ============= CORE XỬ LÝ AI & CAMERA =============
class VideoStreamThread:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
    def update(self):
        while not self.stopped:
            if not self.grabbed: self.stop()
            else: (self.grabbed, self.frame) = self.stream.read()
    def read(self): return self.frame
    def stop(self):
        self.stopped = True
        self.stream.release()

def extract_raw_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[11:23]]).flatten() if results.pose_landmarks else np.zeros(12*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def calculate_motion_score(buffer):
    if len(buffer) < 2: return 0.0
    hands_only = np.array(buffer)[:, 36:] 
    return np.mean(np.sum(np.abs(np.diff(hands_only, axis=0)), axis=1))

def run_realtime_esp32():
    """Luồng xử lý Video độc lập"""
    print("Đang tải Model LSTM...")
    model = load_model(MODEL_PATH)
    train_mean = np.load(MEAN_PATH)
    train_std = np.load(STD_PATH)
    
    print(f"Đang kết nối Camera: {ESP32_URL} ...")
    stream_thread = VideoStreamThread(ESP32_URL).start()
    time.sleep(2)
    
    mp_holistic = mp.solutions.holistic
    motion_buffer = deque(maxlen=5) 
    action_frames = []              
    is_recording = False            
    sentence = []
    
    last_action_time = time.time()
    PAUSE_THRESHOLD = 3.0 # Đứng im 3s là tự cắt câu đi dịch
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            frame = stream_thread.read()
            if frame is None: continue
            frame = cv2.resize(frame, (640, 480))
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            raw_keypoints = extract_raw_keypoints(results)
            motion_buffer.append(raw_keypoints)
            current_motion = calculate_motion_score(list(motion_buffer)) if len(motion_buffer) == 5 else 0
            
            # --- START/STOP LOGIC ---
            if not is_recording:
                if current_motion > 0.04:
                    is_recording = True
                    action_frames = list(motion_buffer)
                    # Báo cho Web biết đang múa để chớp đèn đỏ
                    if main_loop: asyncio.run_coroutine_threadsafe(broadcast_ws({"event": "recording_start"}), main_loop)
            else:
                action_frames.append(raw_keypoints)
                cv2.rectangle(image, (5, 5), (635, 475), (0, 0, 255), 4) # Viền đỏ
                
                if current_motion < 0.02 or len(action_frames) > 100:
                    is_recording = False
                    if main_loop: asyncio.run_coroutine_threadsafe(broadcast_ws({"event": "recording_stop"}), main_loop)
                    
                    if len(action_frames) >= 15:
                        features = process_single_video_features(action_frames)
                        if features is not None:
                            features_norm = (features - train_mean) / (train_std + 1e-7)
                            preds = model.predict(np.expand_dims(features_norm, axis=0), verbose=0)[0]
                            best_idx = np.argmax(preds)
                            best_conf = preds[best_idx]
                            
                            if best_conf > 0.6:
                                current_action = ACTIONS[best_idx]
                                print(f"> Phát hiện: {current_action}")
                                
                                if not sentence or sentence[-1] != current_action:
                                    sentence.append(current_action)
                                    last_action_time = time.time()
                                    
                                    # Đẩy ngay 1 từ vựng mới lên Web
                                    if main_loop:
                                        asyncio.run_coroutine_threadsafe(broadcast_ws({
                                            "event": "word_detected",
                                            "action": current_action,
                                            "confidence": float(best_conf),
                                            "keywords": sentence
                                        }), main_loop)
                    action_frames = [] 

            # --- AUTO TRANSLATE LOGIC ---
            if not is_recording and len(sentence) > 0:
                if time.time() - last_action_time > PAUSE_THRESHOLD:
                    # Đẩy câu đi dịch
                    if main_loop:
                        asyncio.run_coroutine_threadsafe(translate_and_broadcast(sentence.copy()), main_loop)
                    sentence.clear()

            cv2.imshow("Server AI Camera", image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    stream_thread.stop()
    cv2.destroyAllWindows()

# Khởi chạy luồng Camera khi bật Server
@app.on_event("startup")
def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    threading.Thread(target=run_realtime_esp32, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run("server_esp32:app", host="0.0.0.0", port=8000, reload=True)