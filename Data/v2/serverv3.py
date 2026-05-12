import cv2
import numpy as np
import mediapipe as mp
import threading
import asyncio
import time
import os
import re
import queue
import urllib.parse
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from featurev2 import process_single_video_features

import requests
from gtts import gTTS
from fastapi.staticfiles import StaticFiles

# ============= CẤU HÌNH =============
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

ESP32_IP    = "192.168.1.26"
ESP32_PORT  = 82
SERVER_IP   = "192.168.2.76"
SERVER_PORT = 8000
TTS_DIR     = os.path.join(BASE_DIR, "tts_cache")
os.makedirs(TTS_DIR, exist_ok=True)

# ESP32_URL            = "http://10.10.49.160:81/stream"
ESP32_URL            = 0
CONFIDENCE_THRESHOLD = 0.5
START_THRESHOLD      = 0.065
STOP_THRESHOLD       = 0.04
MIN_ACTION_FRAMES    = 18
PAUSE_THRESHOLD      = 5.0
STOP_PATIENCE_FRAMES = 7

# ============= TTS + LCD =============
def speak_on_esp32(sentence: str):
    try:
        safe_text = urllib.parse.quote(sentence)
        lcd_url   = f"http://{ESP32_IP}:{ESP32_PORT}/display?text={safe_text}"
        try:
            requests.get(lcd_url, timeout=2)
            print(f"[LCD] Đã bắn chữ: {sentence}")
        except Exception as e:
            print(f"[LCD ERROR] {e}")

        mp3_path = os.path.join(TTS_DIR, "latest.mp3")
        tts = gTTS(text=sentence, lang='en')
        tts.save(mp3_path)
        print(f"[TTS] Đã tạo file: {mp3_path}")

        audio_url = f"http://{SERVER_IP}:{SERVER_PORT}/tts/latest.mp3"
        esp32_url = f"http://{ESP32_IP}:{ESP32_PORT}/play?url={audio_url}"
        resp = requests.get(esp32_url, timeout=5)
        print(f"[TTS] ESP32 response: {resp.text}")
    except Exception as e:
        print(f"[TTS ERROR] {e}")

def speak_async(sentence: str):
    threading.Thread(target=speak_on_esp32, args=(sentence,), daemon=True).start()

# ============= NHÃN & HÀNH ĐỘNG =============
def load_actions_from_folders():
    data_dir = os.path.join(PROJECT_DIR, 'npy_datav2')
    if not os.path.exists(data_dir): return []
    return sorted(
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith('.')
    )

ACTIONS = load_actions_from_folders()

KEYWORD_ALIAS = {
    "father - dad": "father", "mother - mom": "mother", "toilet-wc": "toilet",
    "home-house":   "home",   "thankyou": "thank you",  "hunggry":  "hungry",
    "tomorow": "tomorrow",    "canccel":  "cancel",
}

def normalize_keyword(raw_label: str) -> str:
    s = raw_label.strip().lower()
    if s in KEYWORD_ALIAS: return KEYWORD_ALIAS[s]
    s = re.sub(r"[\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return KEYWORD_ALIAS.get(s, s)

def apply_sentence_action(sentence: list[str], action: str) -> tuple[list[str], str | None]:
    if action == "cancel":
        if sentence: return sentence, sentence.pop()
        return sentence, None
    if not sentence or sentence[-1] != action:
        sentence.append(action)
    return sentence, None

# ============= TẢI MÔ HÌNH =============
MODEL_PATH = os.path.join(PROJECT_DIR, 'model', f'hybrid_model_{len(ACTIONS)}tu.keras')
MEAN_PATH  = os.path.join(PROJECT_DIR, 'model', 'train_mean.npy')
STD_PATH   = os.path.join(PROJECT_DIR, 'model', 'train_std.npy')

def load_best_checkpoint(model_dir):
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
    if checkpoints:
        checkpoints.sort()
        return os.path.join(model_dir, checkpoints[-1])
    return model_dir

NLG_MODEL_PATH = os.path.join(PROJECT_DIR, 'model', 't5_sign_model')
best_ckpt      = load_best_checkpoint(NLG_MODEL_PATH)

print("\n[NLP] Đang tải não bộ dịch câu T5 Local...")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(best_ckpt)
nlg_model = T5ForConditionalGeneration.from_pretrained(best_ckpt).to(device)
print(f"[NLP] Tải thành công! Đang chạy trên: {device}\n")

# ============= WEBSOCKET =============
connected_clients: set[WebSocket] = set()
clients_lock = threading.Lock()
_main_loop: asyncio.AbstractEventLoop | None = None

async def _broadcast(message: dict):
    dead = set()
    with clients_lock: targets = set(connected_clients)
    for ws in targets:
        try:
            await ws.send_json(message)
        except:
            dead.add(ws)
    if dead:
        with clients_lock: connected_clients.difference_update(dead)

def broadcast(message: dict):
    if _main_loop and not _main_loop.is_closed():
        asyncio.run_coroutine_threadsafe(_broadcast(message), _main_loop)

# ============= NLG WORKER =============
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

# ============= [THREAD 1] VIDEO STREAM =============
class VideoStreamThread:
    """
    Chỉ làm 1 việc: đọc frame từ ESP32 và để sẵn.
    Không resize, không xử lý gì thêm.
    """
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        grabbed, frame = self.cap.read()
        self.frame   = frame if grabbed else None
        self.grabbed = grabbed
        self.stopped = False
        self._lock   = threading.Lock()

    def start(self):
        threading.Thread(target=self._update, daemon=True, name="VideoReader").start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self._lock:
                    self.frame   = frame
                    self.grabbed = True
            else:
                # [FIX] Nhường CPU khi không có frame
                time.sleep(0.01)
                with self._lock:
                    self.grabbed = False

    def read(self):
        with self._lock:
            return self.frame

    def stop(self):
        self.stopped = True
        if self.cap: self.cap.release()


# ============= SHARED DISPLAY STATE =============
class DisplayState:
    """
    Object chia sẻ giữa HolisticWorker và Main thread.
    Main thread chỉ đọc để hiển thị, không xử lý logic.
    """
    def __init__(self):
        self._lock         = threading.Lock()
        self.annotated     = None   # frame đã vẽ landmarks + border
        self.motion        = 0.0
        self.is_recording  = False
        self.action        = None
        self.conf          = 0.0
        self.sentence      = []
        self.fps           = 0.0

    def set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(
                annotated    = self.annotated,
                motion       = self.motion,
                is_recording = self.is_recording,
                action       = self.action,
                conf         = self.conf,
                sentence     = list(self.sentence),
                fps          = self.fps,
            )


# ============= KEYPOINT HELPERS =============
FACE_INDICES = [0, 13, 14, 17, 61, 291, 39, 181, 269, 405,
                46, 52, 65, 276, 282, 295, 33, 133, 362, 263]

def extract_raw_keypoints(results) -> np.ndarray:
    pose = (np.array([[lm.x, lm.y, lm.z]
                      for lm in results.pose_landmarks.landmark[11:23]]).flatten()
            if results.pose_landmarks else np.zeros(12 * 3))
    lh   = (np.array([[lm.x, lm.y, lm.z]
                      for lm in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21 * 3))
    rh   = (np.array([[lm.x, lm.y, lm.z]
                      for lm in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21 * 3))
    face = (np.array([[results.face_landmarks.landmark[i].x,
                       results.face_landmarks.landmark[i].y,
                       results.face_landmarks.landmark[i].z]
                      for i in FACE_INDICES]).flatten()
            if results.face_landmarks else np.zeros(20 * 3))
    return np.concatenate([pose, lh, rh, face])

def calculate_motion_score(buffer: list) -> float:
    if len(buffer) < 2: return 0.0
    seq   = np.array(buffer)
    hands = seq[:, 36:162]
    diffs = np.abs(np.diff(hands, axis=0))
    return float(np.mean(np.sum(diffs, axis=1)))


# ============= [THREAD 3] INFERENCE WORKER =============
# Queue kích thước 1: action mới đến khi worker còn bận → tự bỏ action cũ
_infer_queue: queue.Queue = queue.Queue(maxsize=1)

def _inference_worker(model, train_mean, train_std):
    """
    Chạy model.predict hoàn toàn độc lập.
    Không block camera, không block display.
    """
    while True:
        task = _infer_queue.get()
        if task is None:        # sentinel → thoát
            break

        valid_frames, callback = task
        try:
            features = process_single_video_features(valid_frames)
            if features is None:
                callback(None)
                continue

            feat_norm = (features - train_mean) / (train_std + 1e-7)
            X         = np.expand_dims(feat_norm, axis=0)
            preds     = model.predict(X, verbose=0)[0]

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


# ============= [THREAD 2] HOLISTIC WORKER =============
def _holistic_worker(stream: VideoStreamThread,
                     display_state: DisplayState,
                     train_mean: np.ndarray,
                     train_std:  np.ndarray):
    """
    Thread nặng nhất: chạy holistic.process() + state machine.
    Tách hoàn toàn khỏi main thread → main thread không bao giờ bị block.
    """
    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils

    motion_buffer          = deque(maxlen=5)
    action_frames: list    = []
    is_recording           = False
    frames_below_threshold = 0
    sentence: list[str]    = []
    current_action         = None
    display_conf           = 0.0
    last_action_time       = time.time()
    fps_buf                = deque(maxlen=30)
    prev_time              = time.time()
    is_inferring           = False

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while True:
            frame = stream.read()
            if frame is None:
                time.sleep(0.005)
                continue

            # ── MediaPipe ────────────────────────────────────────────────
            # Resize ở đây thay vì VideoStreamThread để giữ thread đó gọn
            frame   = cv2.resize(frame, (640, 480))
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)          # ← phần nặng nhất
            rgb.flags.writeable = True
            image   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Chỉ vẽ tay, bỏ mặt và pose
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )

            # ── Keypoints & motion ───────────────────────────────────────
            raw_kp = extract_raw_keypoints(results)
            motion_buffer.append(raw_kp)
            current_motion = (
                calculate_motion_score(list(motion_buffer))
                if len(motion_buffer) == 5 else 0.0
            )

            # ── Máy trạng thái ───────────────────────────────────────────
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

                stop_cond = (
                    frames_below_threshold >= STOP_PATIENCE_FRAMES
                    or len(action_frames) > 100
                )

                if stop_cond:
                    is_recording = False
                    broadcast({"event": "recording_stop"})

                    valid_frames = (
                        action_frames[:-frames_below_threshold]
                        if frames_below_threshold > 0 else action_frames
                    )

                    if len(valid_frames) >= MIN_ACTION_FRAMES and not is_inferring:
                        # Snapshot sentence tại thời điểm này để callback dùng đúng
                        _sentence_snap = sentence.copy()

                        def on_infer_done(result,
                                          _snap=_sentence_snap):
                            nonlocal current_action, display_conf, last_action_time
                            nonlocal is_inferring, sentence
                            is_inferring = False

                            if result is None:
                                broadcast({"event": "action_too_short"})
                                return

                            best_idx, best_conf, top3_results = result
                            if best_conf > CONFIDENCE_THRESHOLD:
                                current_action    = ACTIONS[best_idx]
                                normalized_action = normalize_keyword(current_action)
                                display_conf      = best_conf

                                sentence, deleted_word = apply_sentence_action(
                                    sentence, normalized_action
                                )
                                last_action_time = time.time()

                                if normalized_action == "cancel":
                                    broadcast({
                                        "event":             "cancel_detected",
                                        "action":            current_action,
                                        "action_normalized": normalized_action,
                                        "deleted_word":      deleted_word,
                                        "confidence":        best_conf,
                                        "top3":              top3_results,
                                        "keywords":          list(sentence),
                                    })
                                else:
                                    broadcast({
                                        "event":             "word_detected",
                                        "action":            current_action,
                                        "action_normalized": normalized_action,
                                        "confidence":        best_conf,
                                        "top3":              top3_results,
                                        "keywords":          list(sentence),
                                    })

                            # Cập nhật display sau khi inference xong
                            display_state.set(
                                action   = current_action,
                                conf     = display_conf,
                                sentence = list(sentence),
                            )

                        try:
                            # put_nowait: không block, bỏ qua nếu queue đầy
                            _infer_queue.put_nowait((valid_frames.copy(), on_infer_done))
                            is_inferring = True
                        except queue.Full:
                            print("[HOLISTIC] Inference đang bận, bỏ action này")

                    elif len(valid_frames) < MIN_ACTION_FRAMES:
                        broadcast({"event": "action_too_short"})

                    action_frames = []

            # ── Auto translate ────────────────────────────────────────────
            if (
                not is_recording
                and sentence
                and time.time() - last_action_time > PAUSE_THRESHOLD
            ):
                call_nlg_async(sentence.copy())
                sentence.clear()
                current_action = None
                display_state.set(action=None, conf=0.0, sentence=[])

            # ── FPS ──────────────────────────────────────────────────────
            now = time.time()
            fps_buf.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # ── Đẩy frame đã annotate lên DisplayState ───────────────────
            # Main thread lấy ra imshow mà không cần biết gì về logic
            display_state.set(
                annotated    = image,
                motion       = current_motion,
                is_recording = is_recording,
                action       = current_action if not is_recording else None,
                conf         = display_conf   if not is_recording else 0.0,
                sentence     = list(sentence),
                fps          = float(np.mean(fps_buf)),
            )


# ============= SIDEBAR =============
def draw_sidebar(fps, motion, is_recording, current_action, confidence, sentence) -> np.ndarray:
    sb = np.zeros((480, 320, 3), dtype=np.uint8)
    sb[:] = (20, 20, 20)
    font  = cv2.FONT_HERSHEY_SIMPLEX

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


# ============= MAIN CAMERA ENTRY =============
def run_camera():
    print("[CAM] Đang tải model LSTM...")
    model      = load_model(MODEL_PATH)
    train_mean = np.load(MEAN_PATH)
    train_std  = np.load(STD_PATH)

    # [Thread 3] Inference worker
    threading.Thread(
        target=_inference_worker,
        args=(model, train_mean, train_std),
        daemon=True,
        name="InferenceWorker"
    ).start()
    print("[CAM] InferenceWorker started.")

    # [Thread 1] Video stream
    stream = VideoStreamThread(ESP32_URL).start()
    time.sleep(2.0)

    # Shared state giữa HolisticWorker và Main thread
    display_state = DisplayState()

    # [Thread 2] Holistic worker
    threading.Thread(
        target=_holistic_worker,
        args=(stream, display_state, train_mean, train_std),
        daemon=True,
        name="HolisticWorker"
    ).start()
    print("[CAM] HolisticWorker started.")

    # ── [Main thread] Chỉ display, không bao giờ block ───────────────────
    # imshow PHẢI chạy trên main thread (yêu cầu của OpenCV GUI)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    while True:
        snap = display_state.snapshot()

        frame = snap["annotated"] if snap["annotated"] is not None else blank
        sidebar = draw_sidebar(
            snap["fps"],
            snap["motion"],
            snap["is_recording"],
            snap["action"],
            snap["conf"],
            snap["sentence"],
        )
        display = np.hstack((frame, sidebar))
        cv2.imshow("Sign Language AI", display)

        # waitKey(1): nhả ~1ms cho GUI event loop, không block logic
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp
    _infer_queue.put(None)   # sentinel dừng InferenceWorker
    stream.stop()
    cv2.destroyAllWindows()
    print("[CAM] Đã dừng.")


# ============= FASTAPI =============
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    threading.Thread(target=run_camera, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/tts", StaticFiles(directory=TTS_DIR), name="tts")

@app.websocket("/ws/sign-language")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    with clients_lock:
        connected_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(30)
    except:
        pass
    finally:
        with clients_lock:
            connected_clients.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)