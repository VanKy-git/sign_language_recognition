import cv2
import numpy as np
import mediapipe as mp
import threading
import asyncio
import time
import os
import re
import urllib.parse
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# Import thư viện cho AI Local (T5)
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Lõi xử lý đặc trưng 504 features
from featurev2 import process_single_video_features

# ============= CẤU HÌNH BỔ SUNG =============
import requests
from gtts import gTTS
from fastapi.staticfiles import StaticFiles
import queue

# Queue kích thước 1: nếu inference chưa xong mà có action mới → bỏ action cũ
# (tránh tích lũy hàng đợi dài khi người dùng ký liên tục nhanh)
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR     = os.path.dirname(BASE_DIR)

# Nhớ kiểm tra lại IP trước khi chạy
ESP32_IP        = "10.10.49.144"   
ESP32_PORT      = 82
SERVER_IP       = "10.10.49.146"   
SERVER_PORT     = 8000
TTS_DIR         = os.path.join(BASE_DIR, "tts_cache")
os.makedirs(TTS_DIR, exist_ok=True)

# ============= HÀM TTS + LCD + GỌI ESP32 =============
def speak_on_esp32(sentence: str):
    """Bắn chữ lên LCD -> Tạo file MP3 -> Gọi ESP32 phát âm thanh tiếng Anh"""
    try:
        # 1. Bắn chữ lên màn hình LCD trước
        safe_text = urllib.parse.quote(sentence)
        lcd_url = f"http://{ESP32_IP}:{ESP32_PORT}/display?text={safe_text}"
        try:
            requests.get(lcd_url, timeout=2)
            print(f"[LCD] Đã bắn chữ: {sentence}")
        except Exception as e:
            print(f"[LCD ERROR] {e}")

        # 2. Tạo file MP3 (TIẾNG ANH)
        mp3_path = os.path.join(TTS_DIR, "latest.mp3")
        tts = gTTS(text=sentence, lang='en')   
        tts.save(mp3_path)
        print(f"[TTS] Đã tạo file: {mp3_path}")

        # 3. Gọi ESP32 kéo file về phát
        audio_url = f"http://{SERVER_IP}:{SERVER_PORT}/tts/latest.mp3"
        esp32_url = f"http://{ESP32_IP}:{ESP32_PORT}/play?url={audio_url}"
        resp = requests.get(esp32_url, timeout=5)
        print(f"[TTS] ESP32 response: {resp.text}")

    except Exception as e:
        print(f"[TTS ERROR] {e}")

def speak_async(sentence: str):
    threading.Thread(target=speak_on_esp32, args=(sentence,), daemon=True).start()

# ============= XỬ LÝ NHÃN VÀ HÀNH ĐỘNG =============
def load_actions_from_folders():
    data_dir = os.path.join(PROJECT_DIR, 'npy_datav2')
    if not os.path.exists(data_dir): return []
    return sorted(name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith('.'))

ACTIONS = load_actions_from_folders()

KEYWORD_ALIAS = {
    "father - dad": "father", "mother - mom": "mother", "toilet-wc": "toilet",
    "home-house": "home", "thankyou": "thank you", "hunggry": "hungry",
    "tomorow": "tomorrow", "canccel": "cancel",
}

def normalize_keyword(raw_label: str) -> str:
    s = raw_label.strip().lower()
    if s in KEYWORD_ALIAS: return KEYWORD_ALIAS[s]
    s = s.replace("-", " ").replace("_", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return KEYWORD_ALIAS.get(s, s)

def apply_sentence_action(sentence: list[str], action: str) -> tuple[list[str], str | None]:
    if action == "cancel":
        if sentence: return sentence, sentence.pop()
        return sentence, None
    if not sentence or sentence[-1] != action: sentence.append(action)
    return sentence, None

# ============= TẢI MÔ HÌNH =============
MODEL_PATH      = os.path.join(PROJECT_DIR, 'model', f'hybrid_model_{len(ACTIONS)}tu.keras')
MEAN_PATH       = os.path.join(PROJECT_DIR, 'model', 'train_mean.npy')
STD_PATH        = os.path.join(PROJECT_DIR, 'model', 'train_std.npy')

def load_best_checkpoint(model_dir):
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
    if checkpoints:
        checkpoints.sort()
        return os.path.join(model_dir, checkpoints[-1])
    return model_dir

NLG_MODEL_PATH = os.path.join(PROJECT_DIR, 'model', 't5_sign_model')
best_ckpt = load_best_checkpoint(NLG_MODEL_PATH)

print("\n[NLP] Đang tải não bộ dịch câu T5 Local...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(best_ckpt)
nlg_model = T5ForConditionalGeneration.from_pretrained(best_ckpt).to(device)
print(f"[NLP] Tải thành công! Đang chạy trên: {device}\n")

# ============= CẤU HÌNH CAMERA & NGƯỠNG =============
ESP32_URL            = "http://10.10.49.160:81/stream"    
CONFIDENCE_THRESHOLD = 0.5
START_THRESHOLD      = 0.065
STOP_THRESHOLD       = 0.04
MIN_ACTION_FRAMES    = 18
PAUSE_THRESHOLD      = 5.0   
STOP_PATIENCE_FRAMES = 7     

# ============= WEBSOCKET & NLP WORKER =============
connected_clients: set[WebSocket] = set()
clients_lock = threading.Lock()
_main_loop: asyncio.AbstractEventLoop | None = None

async def _broadcast(message: dict):
    dead = set()
    with clients_lock: targets = set(connected_clients)
    for ws in targets:
        try: await ws.send_json(message)
        except: dead.add(ws)
    if dead:
        with clients_lock: connected_clients.difference_update(dead)

def broadcast(message: dict):
    if _main_loop and not _main_loop.is_closed():
        asyncio.run_coroutine_threadsafe(_broadcast(message), _main_loop)

def _local_nlg_worker(keywords: list[str]):
    if not keywords: return
    model_keywords = keywords.copy()
    if model_keywords[0].lower() == "me":
        model_keywords[0] = "I"
    kw_string  = " ".join(model_keywords)
    input_text = f"keywords to sentence: {kw_string}"
    print(f"\n[AI TRANSLATING...] Input: {input_text}")
    try:
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outputs = nlg_model.generate(input_ids, max_length=64, num_beams=4)
        natural_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 KẾT QUẢ DỊCH: {natural_sentence}\n")
        broadcast({"event": "translation_success", "natural_sentence": natural_sentence})
        speak_async(natural_sentence)   
    except Exception as exc:
        print(f"[LỖI NLG LOCAL] {exc}")
        fallback = kw_string
        broadcast({"event": "translation_success", "natural_sentence": fallback})
        speak_async(fallback)           

def call_nlg_async(keywords: list[str]):
    threading.Thread(target=_local_nlg_worker, args=(keywords,), daemon=True).start()

# ===== Thêm inference worker (đặt sau hàm call_nlg_async) =====
def _inference_worker(model, train_mean, train_std):
    """Thread riêng xử lý model.predict, không block camera loop."""
    while True:
        task = _infer_queue.get()          # block chờ data
        if task is None:                    # sentinel để dừng thread
            break

        valid_frames, callback = task
        try:
            features = process_single_video_features(valid_frames)
            if features is None:
                callback(None)
                continue

            feat_norm = (features - train_mean) / (train_std + 1e-7)
            X = np.expand_dims(feat_norm, axis=0)
            preds = model.predict(X, verbose=0)[0]   # ← chỉ chạy ở đây

            top3_indices = np.argsort(preds)[-3:][::-1]
            top3_results = [{"word": ACTIONS[i], "confidence": float(preds[i])}
                            for i in top3_indices]
            best_idx  = int(top3_indices[0])
            best_conf = float(preds[best_idx])
            callback((best_idx, best_conf, top3_results))

        except Exception as e:
            print(f"[INFER ERROR] {e}")
            callback(None)
        finally:
            _infer_queue.task_done()

# ============= CAMERA BACKGROUND THREAD (ĐÃ TỐI ƯU RESIZE) =============
# class VideoStreamThread:
#     def __init__(self, src):
#         self.cap = cv2.VideoCapture(src)
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         self.grabbed, frame = self.cap.read()
#         # Ép khung hình về 640x480 ngay từ luồng phụ để cứu CPU
#         self.frame = cv2.resize(frame, (640, 480)) if self.grabbed else None
#         self.stopped = False

#     def start(self):
#         threading.Thread(target=self._update, daemon=True).start()
#         return self

#     # def _update(self):
#     #     while not self.stopped:
#     #         grabbed, frame = self.cap.read()
#     #         if grabbed:
#     #             # Thu nhỏ ảnh ngầm định tại đây
#     #             self.frame = cv2.resize(frame, (640, 480))
#     #         self.grabbed = grabbed

#     def _update(self):
#         while not self.stopped:
#             grabbed, frame = self.cap.read()
#             if grabbed:
#                 self.frame = cv2.resize(frame, (640, 480))
#                 self.grabbed = True
#             else:
#                 # Không có frame → nhường CPU, tránh busy spin
#                 # 10ms đủ để không bỏ lỡ frame (stream ~15-20fps = 50-66ms/frame)
#                 time.sleep(0.01)
#                 self.grabbed = False
#     def read(self): return self.frame
    
#     def stop(self): 
#         self.stopped = True
#         if self.cap: self.cap.release()

# ============= CÀO DỮ LIỆU & ĐO VẬN TỐC =============
FACE_INDICES = [0, 13, 14, 17, 61, 291, 39, 181, 269, 405, 46, 52, 65, 276, 282, 295, 33, 133, 362, 263]

def extract_raw_keypoints(results) -> np.ndarray:
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark[11:23]]).flatten() if results.pose_landmarks else np.zeros(12 * 3)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    face = np.array([[results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z] for i in FACE_INDICES]).flatten() if results.face_landmarks else np.zeros(20 * 3)
    return np.concatenate([pose, lh, rh, face])

def calculate_motion_score(buffer: list) -> float:
    if len(buffer) < 2: return 0.0
    seq = np.array(buffer)
    hands = seq[:, 36:162]                          
    diffs = np.abs(np.diff(hands, axis=0))
    return float(np.mean(np.sum(diffs, axis=1)))

# ============= GIAO DIỆN HIỂN THỊ (THU GỌN VỀ 480P) =============
def draw_sidebar(fps, motion, is_recording, current_action, confidence, sentence) -> np.ndarray:
    sb = np.zeros((480, 320, 3), dtype=np.uint8) # Khớp với chiều cao 480 của Camera
    sb[:] = (20, 20, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(sb, (0, 0), (320, 45), (40, 40, 80), -1)
    cv2.putText(sb, "SIGN LANGUAGE AI", (10, 30), font, 0.6, (180, 180, 255), 2)
    cv2.putText(sb, f"FPS: {fps:>5.1f}   Motion: {motion:.3f}", (10, 70), font, 0.5, (100, 200, 100), 1)

    cv2.putText(sb, "PREDICTION", (10, 110), font, 0.55, (200, 200, 200), 1)
    cv2.line(sb, (10, 118), (310, 118), (60, 60, 60), 1)

    if is_recording:
        cv2.rectangle(sb, (10, 128), (310, 210), (0, 0, 120), -1)
        cv2.putText(sb, "Recording...", (20, 175), font, 0.8, (100, 150, 255), 2)
    elif current_action:
        cv2.rectangle(sb, (10, 128), (310, 210), (0, 80, 0), -1)
        scale = max(0.6, 1.2 - len(current_action) * 0.03)
        cv2.putText(sb, current_action, (20, 175), font, scale, (255, 255, 255), 2)
        cv2.putText(sb, f"Conf: {confidence * 100:.1f}%", (20, 200), font, 0.5, (180, 255, 180), 1)
    else:
        cv2.rectangle(sb, (10, 128), (310, 210), (40, 40, 40), -1)

    cv2.putText(sb, "SENTENCE", (10, 240), font, 0.55, (200, 200, 200), 1)
    cv2.line(sb, (10, 248), (310, 248), (60, 60, 60), 1)
    y = 280
    for word in reversed(sentence):
        cv2.putText(sb, f"> {word}", (15, y), font, 0.7, (160, 210, 255), 2)
        y += 35
        if y > 470: break

    return sb

# ============= MAIN LOGIC =============
# def run_camera():
#     print("[CAM] Đang tải model Nhận diện hình ảnh (LSTM)...")
#     model      = load_model(MODEL_PATH)
#     train_mean = np.load(MEAN_PATH)
#     train_std  = np.load(STD_PATH)

#     stream = VideoStreamThread(ESP32_URL).start()
#     time.sleep(2.0)

#     mp_holistic  = mp.solutions.holistic
#     mp_drawing   = mp.solutions.drawing_utils

#     motion_buffer   = deque(maxlen=5)
#     action_frames: list[np.ndarray] = []
#     is_recording    = False
#     frames_below_threshold = 0  
#     sentence: list[str] = []
#     current_action: str | None = None
#     display_conf    = 0.0
#     last_action_time = time.time()
#     fps_buf         = deque(maxlen=30)
#     prev_time       = time.time()

#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while True:
#             frame = stream.read()
#             if frame is None: continue
            
#             # Không cần resize ở đây nữa vì đã làm trong luồng phụ!
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             rgb.flags.writeable = False
#             results = holistic.process(rgb)
#             rgb.flags.writeable = True
#             image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

#             # CHỈ vẽ khung xương tay, TẮT vẽ mặt và người để giảm lag
#             if results.left_hand_landmarks:
#                 mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#             if results.right_hand_landmarks:
#                 mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

#             raw_kp = extract_raw_keypoints(results)
#             motion_buffer.append(raw_kp)
#             current_motion = calculate_motion_score(list(motion_buffer)) if len(motion_buffer) == 5 else 0.0

#             if not is_recording:
#                 if current_motion > START_THRESHOLD:
#                     is_recording = True
#                     action_frames = list(motion_buffer)
#                     frames_below_threshold = 0  
#                     broadcast({"event": "recording_start"})
#             else:
#                 action_frames.append(raw_kp)
#                 # Bọc viền đỏ (khớp với tỷ lệ màn 640x480)
#                 cv2.rectangle(image, (4, 4), (635, 475), (0, 0, 200), 4)

#                 if current_motion < STOP_THRESHOLD:
#                     frames_below_threshold += 1
#                 else:
#                     frames_below_threshold = 0  

#                 if frames_below_threshold >= STOP_PATIENCE_FRAMES or len(action_frames) > 100:
#                     is_recording = False
#                     broadcast({"event": "recording_stop"})

#                     valid_frames = action_frames[:-frames_below_threshold] if frames_below_threshold > 0 else action_frames

#                     if len(valid_frames) >= MIN_ACTION_FRAMES:
#                         features = process_single_video_features(valid_frames)
#                         if features is not None:
#                             feat_norm = (features - train_mean) / (train_std + 1e-7)
#                             X = np.expand_dims(feat_norm, axis=0)
#                             preds = model.predict(X, verbose=0)[0]

#                             top3_indices = np.argsort(preds)[-3:][::-1]
#                             top3_results = [{"word": ACTIONS[i], "confidence": float(preds[i])} for i in top3_indices]

#                             best_idx = int(top3_indices[0])
#                             best_conf = float(preds[best_idx])

#                             if best_conf > CONFIDENCE_THRESHOLD:
#                                 current_action = ACTIONS[best_idx]
#                                 normalized_action = normalize_keyword(current_action)
#                                 display_conf   = best_conf

#                                 sentence, deleted_word = apply_sentence_action(sentence, normalized_action)
#                                 last_action_time = time.time()

#                                 if normalized_action == "cancel":
#                                     broadcast({"event": "cancel_detected", "action": current_action, "action_normalized": normalized_action, "deleted_word": deleted_word, "confidence": best_conf, "top3": top3_results, "keywords": list(sentence)})
#                                 else:
#                                     broadcast({"event": "word_detected", "action": current_action, "action_normalized": normalized_action, "confidence": best_conf, "top3": top3_results, "keywords": list(sentence)})
#                         else: broadcast({"event": "action_too_short"})
#                     action_frames = []

#             if not is_recording and sentence and time.time() - last_action_time > PAUSE_THRESHOLD:
#                 call_nlg_async(sentence.copy())
#                 sentence.clear()
#                 current_action = None

#             now = time.time()
#             fps_buf.append(1.0 / max(now - prev_time, 1e-6))
#             prev_time = now

#             sidebar = draw_sidebar(float(np.mean(fps_buf)), current_motion, is_recording, current_action if not is_recording else None, display_conf if not is_recording else 0.0, sentence)
#             display = np.hstack((image, sidebar))
#             cv2.imshow("Sign Language AI", display)

#             if cv2.waitKey(1) & 0xFF == ord('q'): break

#     stream.stop()
#     cv2.destroyAllWindows()


# ============= CAMERA BACKGROUND THREAD =============
class VideoStreamThread:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, frame = self.cap.read()
        self.frame = cv2.resize(frame, (640, 480)) if self.grabbed else None
        self.stopped = False

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if grabbed:
                self.frame = cv2.resize(frame, (640, 480))
                self.grabbed = True
            else:
                # [FIX 1] Nhường CPU khi không có frame, tránh busy spin
                time.sleep(0.01)
                self.grabbed = False

    def read(self): return self.frame

    def stop(self):
        self.stopped = True
        if self.cap: self.cap.release()


# ============= [FIX 4] INFERENCE WORKER (TÁCH RIÊNG KHỎI CAMERA LOOP) =============
# Queue kích thước 1: nếu inference chưa xong mà có action mới → tự động bỏ action cũ
# tránh tích lũy hàng đợi dài khi ký nhanh liên tiếp
_infer_queue: queue.Queue = queue.Queue(maxsize=1)


def _inference_worker(model, train_mean, train_std):
    """
    Thread riêng biệt, chỉ làm nhiệm vụ inference.
    Camera loop KHÔNG bị block khi model.predict() đang chạy.
    """
    while True:
        task = _infer_queue.get()       # block, chờ data từ camera loop
        if task is None:                # sentinel value → thoát thread
            break

        valid_frames, callback = task
        try:
            features = process_single_video_features(valid_frames)
            if features is None:
                callback(None)
                continue

            feat_norm = (features - train_mean) / (train_std + 1e-7)
            X         = np.expand_dims(feat_norm, axis=0)
            preds     = model.predict(X, verbose=0)[0]   # ← nặng, nhưng chạy ở đây thôi

            top3_indices = np.argsort(preds)[-3:][::-1]
            top3_results = [
                {"word": ACTIONS[i], "confidence": float(preds[i])}
                for i in top3_indices
            ]
            best_idx  = int(top3_indices[0])
            best_conf = float(preds[best_idx])
            callback((best_idx, best_conf, top3_results))

        except Exception as e:
            print(f"[INFER ERROR] {e}")
            callback(None)
        finally:
            _infer_queue.task_done()


# ============= MAIN LOGIC =============
def run_camera():
    print("[CAM] Đang tải model Nhận diện hình ảnh (LSTM)...")
    model      = load_model(MODEL_PATH)
    train_mean = np.load(MEAN_PATH)
    train_std  = np.load(STD_PATH)

    # [FIX 4] Khởi động inference thread trước khi vào camera loop
    infer_thread = threading.Thread(
        target=_inference_worker,
        args=(model, train_mean, train_std),
        daemon=True,
        name="InferenceWorker"
    )
    infer_thread.start()
    print("[CAM] Inference worker đã khởi động.")

    stream = VideoStreamThread(ESP32_URL).start()
    time.sleep(2.0)

    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils

    motion_buffer          = deque(maxlen=5)
    action_frames: list[np.ndarray] = []
    is_recording           = False
    frames_below_threshold = 0
    sentence: list[str]    = []
    current_action: str | None = None
    display_conf           = 0.0
    last_action_time       = time.time()
    fps_buf                = deque(maxlen=30)
    prev_time              = time.time()

    # [FIX 4] Flag: đang chờ inference worker xử lý, không gửi thêm
    is_inferring = False

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            # Không cần resize ở đây nữa vì đã làm trong _update()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Chỉ vẽ tay, tắt mặt và pose để giảm lag render
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )

            raw_kp = extract_raw_keypoints(results)
            motion_buffer.append(raw_kp)
            current_motion = (
                calculate_motion_score(list(motion_buffer))
                if len(motion_buffer) == 5 else 0.0
            )

            # ── Máy trạng thái: idle / recording ──────────────────────────
            if not is_recording:
                if current_motion > START_THRESHOLD:
                    is_recording           = True
                    action_frames          = list(motion_buffer)
                    frames_below_threshold = 0
                    broadcast({"event": "recording_start"})

            else:
                action_frames.append(raw_kp)
                cv2.rectangle(image, (4, 4), (635, 475), (0, 0, 200), 4)

                if current_motion < STOP_THRESHOLD:
                    frames_below_threshold += 1
                else:
                    frames_below_threshold = 0

                stop_condition = (
                    frames_below_threshold >= STOP_PATIENCE_FRAMES
                    or len(action_frames) > 100
                )

                if stop_condition:
                    is_recording = False
                    broadcast({"event": "recording_stop"})

                    valid_frames = (
                        action_frames[:-frames_below_threshold]
                        if frames_below_threshold > 0
                        else action_frames
                    )

                    # [FIX 4] Đẩy vào queue, KHÔNG block camera loop
                    if len(valid_frames) >= MIN_ACTION_FRAMES and not is_inferring:

                        # Callback này được gọi từ inference thread
                        # → dùng closure để capture đúng sentence/current state
                        def on_infer_done(result, _sentence=sentence):
                            nonlocal current_action, display_conf, last_action_time, is_inferring
                            is_inferring = False

                            if result is None:
                                broadcast({"event": "action_too_short"})
                                return

                            best_idx, best_conf, top3_results = result

                            if best_conf > CONFIDENCE_THRESHOLD:
                                current_action    = ACTIONS[best_idx]
                                normalized_action = normalize_keyword(current_action)
                                display_conf      = best_conf

                                _sentence, deleted_word = apply_sentence_action(
                                    _sentence, normalized_action
                                )
                                last_action_time = time.time()

                                if normalized_action == "cancel":
                                    broadcast({
                                        "event":              "cancel_detected",
                                        "action":             current_action,
                                        "action_normalized":  normalized_action,
                                        "deleted_word":       deleted_word,
                                        "confidence":         best_conf,
                                        "top3":               top3_results,
                                        "keywords":           list(_sentence),
                                    })
                                else:
                                    broadcast({
                                        "event":             "word_detected",
                                        "action":            current_action,
                                        "action_normalized": normalized_action,
                                        "confidence":        best_conf,
                                        "top3":              top3_results,
                                        "keywords":          list(_sentence),
                                    })

                        try:
                            _infer_queue.put_nowait((valid_frames.copy(), on_infer_done))
                            is_inferring = True
                        except queue.Full:
                            # inference worker vẫn đang bận → bỏ qua lần này
                            print("[CAM] Inference đang bận, bỏ action")

                    elif len(valid_frames) < MIN_ACTION_FRAMES:
                        broadcast({"event": "action_too_short"})

                    action_frames = []

            # ── Auto translate khi dừng ký đủ lâu ────────────────────────
            if (
                not is_recording
                and sentence
                and time.time() - last_action_time > PAUSE_THRESHOLD
            ):
                call_nlg_async(sentence.copy())
                sentence.clear()
                current_action = None

            # ── FPS counter ───────────────────────────────────────────────
            now = time.time()
            fps_buf.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # ── Hiển thị ─────────────────────────────────────────────────
            sidebar = draw_sidebar(
                float(np.mean(fps_buf)),
                current_motion,
                is_recording,
                current_action if not is_recording else None,
                display_conf   if not is_recording else 0.0,
                sentence,
            )
            display = np.hstack((image, sidebar))
            cv2.imshow("Sign Language AI", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Dọn dẹp
    _infer_queue.put(None)   # gửi sentinel để inference thread tự thoát
    stream.stop()
    cv2.destroyAllWindows()

# ============= FASTAPI =============
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    threading.Thread(target=run_camera, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/tts", StaticFiles(directory=TTS_DIR), name="tts")

@app.websocket("/ws/sign-language")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    with clients_lock: connected_clients.add(websocket)
    try:
        while True: await asyncio.sleep(30)
    except: pass
    finally:
        with clients_lock: connected_clients.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)