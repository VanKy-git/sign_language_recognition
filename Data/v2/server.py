import cv2
import numpy as np
import mediapipe as mp
import threading
import asyncio
import time
import os
import re
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

# ============= CẤU HÌNH ĐƯỜNG DẪN & TỪ VỰNG =============
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR     = os.path.dirname(BASE_DIR)

def load_actions_from_folders():
    data_dir = os.path.join(PROJECT_DIR, 'npy_datav2')
    if not os.path.exists(data_dir):
        return []
    actions = sorted(
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith('.')
    )
    return actions

ACTIONS = load_actions_from_folders()

# Chuẩn hóa tên nhãn nhận diện trước khi gửi sang T5 để giảm nhiễu do tên folder.
KEYWORD_ALIAS = {
    "father - dad": "father",
    "mother - mom": "mother",
    "toilet-wc": "toilet",
    "home-house": "home",
    "thankyou": "thank you",
    "hunggry": "hungry",
    "tomorow": "tomorrow",
    "canccel": "cancel",
}

def normalize_keyword(raw_label: str) -> str:
    s = raw_label.strip().lower()
    if s in KEYWORD_ALIAS:
        return KEYWORD_ALIAS[s]

    # Chuyển các ký tự phân tách phổ biến về khoảng trắng để T5 dễ hiểu hơn.
    s = s.replace("-", " ").replace("_", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return KEYWORD_ALIAS.get(s, s)

def apply_sentence_action(sentence: list[str], action: str) -> tuple[list[str], str | None]:
    """Apply a recognized action to the current sentence buffer.
    Returns:
        - updated sentence list
        - deleted word when action is cancel, otherwise None
    """
    if action == "cancel":
        if sentence:
            deleted_word = sentence.pop()
            return sentence, deleted_word
        return sentence, None

    if not sentence or sentence[-1] != action:
        sentence.append(action)

    return sentence, None

# Đường dẫn Model LSTM (Nhận diện tay)
MODEL_PATH      = os.path.join(PROJECT_DIR, 'model', f'hybrid_model_{len(ACTIONS)}tu.keras')
MEAN_PATH       = os.path.join(PROJECT_DIR, 'model', 'train_mean.npy')
STD_PATH        = os.path.join(PROJECT_DIR, 'model', 'train_std.npy')

def load_best_checkpoint(model_dir):
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
    if checkpoints:
        checkpoints.sort()
        return os.path.join(model_dir, checkpoints[-1])
    return model_dir  # fallback nếu không có checkpoint

# Đường dẫn Model T5 (Dịch câu Offline)
NLG_MODEL_PATH = os.path.join(PROJECT_DIR, 'model', 't5_sign_model')
best_ckpt = load_best_checkpoint(NLG_MODEL_PATH)

# ============= CẤU HÌNH CAMERA & NGƯỠNG NHẬN DIỆN =============
ESP32_URL            = 0     # <--- Nhớ đổi lại IP (VD: "http://192.168.1.47:81/stream") nếu dùng ESP32-CAM
CONFIDENCE_THRESHOLD = 0.5
START_THRESHOLD      = 0.065
STOP_THRESHOLD       = 0.04
MIN_ACTION_FRAMES    = 18
PAUSE_THRESHOLD      = 5.0   
STOP_PATIENCE_FRAMES = 7     # CƠ CHẾ MỚI: Cho phép mất dấu tối đa 7 frame để không bị ngắt ngang

# ============= TẢI MÔ HÌNH NLP LOCAL =============
print("\n[NLP] Đang tải não bộ dịch câu T5 Local...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(best_ckpt)
nlg_model = T5ForConditionalGeneration.from_pretrained(best_ckpt).to(device)
print(f"[NLP] Tải thành công! Đang chạy trên: {device}\n")

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
    """Hàm chạy ngầm dịch câu bằng model Local T5"""
    if not keywords: return
    
    kw_string = " ".join(keywords)
    input_text = f"keywords to sentence: {kw_string}"
    print(f"\n[AI TRANSLATING...] Input: {input_text}")
    
    try:
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outputs = nlg_model.generate(
                input_ids, 
                max_length=64, 
                num_beams=4
            )
        natural_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"🤖 KẾT QUẢ DỊCH LOCAL: {natural_sentence}\n")
        broadcast({"event": "translation_success", "natural_sentence": natural_sentence})
        
    except Exception as exc:
        print(f"[LỖI NLG LOCAL] {exc}")
        broadcast({"event": "translation_success", "natural_sentence": kw_string})

def call_nlg_async(keywords: list[str]):
    threading.Thread(target=_local_nlg_worker, args=(keywords,), daemon=True).start()

# ============= CAMERA BACKGROUND THREAD =============
class VideoStreamThread:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()
            if not self.grabbed: self.stop()

    def read(self): return self.frame
    
    def stop(self): 
        self.stopped = True
        if self.cap:
            self.cap.release()

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

# ============= GIAO DIỆN HIỂN THỊ =============
def draw_sidebar(fps, motion, is_recording, current_action, confidence, sentence) -> np.ndarray:
    sb = np.zeros((720, 380, 3), dtype=np.uint8)
    sb[:] = (20, 20, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(sb, (0, 0), (380, 55), (40, 40, 80), -1)
    cv2.putText(sb, "SIGN LANGUAGE AI", (15, 35), font, 0.75, (180, 180, 255), 2)
    cv2.putText(sb, f"FPS: {fps:>5.1f}   Motion: {motion:.3f}", (15, 85), font, 0.55, (100, 200, 100), 1)

    cv2.putText(sb, "PREDICTION", (15, 130), font, 0.65, (200, 200, 200), 1)
    cv2.line(sb, (15, 138), (365, 138), (60, 60, 60), 1)

    if is_recording:
        cv2.rectangle(sb, (15, 148), (365, 250), (0, 0, 120), -1)
        cv2.putText(sb, "  Recording...", (25, 210), font, 1.0, (100, 150, 255), 2)
    elif current_action:
        cv2.rectangle(sb, (15, 148), (365, 250), (0, 80, 0), -1)
        scale = max(0.7, 1.5 - len(current_action) * 0.04)
        cv2.putText(sb, current_action, (25, 215), font, scale, (255, 255, 255), 2)
        cv2.putText(sb, f"Conf: {confidence * 100:.1f}%", (25, 242), font, 0.6, (180, 255, 180), 1)
    else:
        cv2.rectangle(sb, (15, 148), (365, 250), (40, 40, 40), -1)

    cv2.putText(sb, "SENTENCE", (15, 290), font, 0.65, (200, 200, 200), 1)
    cv2.line(sb, (15, 298), (365, 298), (60, 60, 60), 1)
    y = 335
    for word in reversed(sentence):
        cv2.putText(sb, f"> {word}", (20, y), font, 0.85, (160, 210, 255), 2)
        y += 42
        if y > 640: break

    return sb

# ============= MAIN LOGIC =============
def run_camera():
    print("[CAM] Đang tải model Nhận diện hình ảnh (LSTM)...")
    model      = load_model(MODEL_PATH)
    train_mean = np.load(MEAN_PATH)
    train_std  = np.load(STD_PATH)

    stream = VideoStreamThread(ESP32_URL).start()
    time.sleep(2.0)

    mp_holistic  = mp.solutions.holistic
    mp_drawing   = mp.solutions.drawing_utils

    motion_buffer   = deque(maxlen=5)
    action_frames: list[np.ndarray] = []
    is_recording    = False
    frames_below_threshold = 0  # Biến đếm số frame bị rớt tay
    sentence: list[str] = []
    current_action: str | None = None
    display_conf    = 0.0
    last_action_time = time.time()
    fps_buf         = deque(maxlen=30)
    prev_time       = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            frame = stream.read()
            if frame is None: continue
            try: frame = cv2.resize(frame, (960, 720))
            except: continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if results.face_landmarks:
                h, w = image.shape[:2]
                for idx in FACE_INDICES:
                    lm = results.face_landmarks.landmark[idx]
                    cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

            if results.pose_landmarks:
                h, w = image.shape[:2]
                for idx in range(11, 23):
                    lm = results.pose_landmarks.landmark[idx]
                    cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 5, (0, 80, 255), -1)

            raw_kp = extract_raw_keypoints(results)
            motion_buffer.append(raw_kp)
            current_motion = calculate_motion_score(list(motion_buffer)) if len(motion_buffer) == 5 else 0.0

            if not is_recording:
                if current_motion > START_THRESHOLD:
                    is_recording = True
                    action_frames = list(motion_buffer)
                    frames_below_threshold = 0  # Reset bộ đếm khi bắt đầu múa
                    broadcast({"event": "recording_start"})
            else:
                action_frames.append(raw_kp)
                cv2.rectangle(image, (4, 4), (955, 715), (0, 0, 200), 4)

                # --- LOGIC MỚI: CHỐNG CHỚP TAY ---
                if current_motion < STOP_THRESHOLD:
                    frames_below_threshold += 1
                else:
                    frames_below_threshold = 0  # Múa lại -> Hủy đếm

                if frames_below_threshold >= STOP_PATIENCE_FRAMES or len(action_frames) > 100:
                    is_recording = False
                    broadcast({"event": "recording_stop"})

                    # Cắt rác (những frame mất dấu ở cuối) để AI không bị nhiễu
                    valid_frames = action_frames[:-frames_below_threshold] if frames_below_threshold > 0 else action_frames

                    if len(valid_frames) >= MIN_ACTION_FRAMES:
                        features = process_single_video_features(valid_frames)
                        if features is not None:
                            feat_norm = (features - train_mean) / (train_std + 1e-7)
                            X = np.expand_dims(feat_norm, axis=0)
                            preds = model.predict(X, verbose=0)[0]

                            top3_indices = np.argsort(preds)[-3:][::-1]
                            top3_results = [{"word": ACTIONS[i], "confidence": float(preds[i])} for i in top3_indices]

                            best_idx = int(top3_indices[0])
                            best_conf = float(preds[best_idx])

                            if best_conf > CONFIDENCE_THRESHOLD:
                                current_action = ACTIONS[best_idx]
                                normalized_action = normalize_keyword(current_action)
                                display_conf   = best_conf

                                sentence, deleted_word = apply_sentence_action(sentence, normalized_action)
                                last_action_time = time.time()

                                if normalized_action == "cancel":
                                    broadcast({
                                        "event": "cancel_detected",
                                        "action": current_action,
                                        "action_normalized": normalized_action,
                                        "deleted_word": deleted_word,
                                        "confidence": best_conf,
                                        "top3": top3_results,
                                        "keywords": list(sentence),
                                    })
                                else:
                                    broadcast({
                                        "event"     : "word_detected",
                                        "action"    : current_action,
                                        "action_normalized": normalized_action,
                                        "confidence": best_conf,
                                        "top3"      : top3_results,
                                        "keywords"  : list(sentence),
                                    })
                        else: broadcast({"event": "action_too_short"})

                    action_frames = []

            if not is_recording and sentence and time.time() - last_action_time > PAUSE_THRESHOLD:
                call_nlg_async(sentence.copy())
                sentence.clear()
                current_action = None

            now = time.time()
            fps_buf.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            sidebar = draw_sidebar(float(np.mean(fps_buf)), current_motion, is_recording, current_action if not is_recording else None, display_conf if not is_recording else 0.0, sentence)
            display = np.hstack((image, sidebar))
            cv2.imshow("Sign Language Recognition", display)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

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