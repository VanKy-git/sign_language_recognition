import os
import time
import socket
import threading
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# ================= CẤU HÌNH ĐỊA CHỈ =================
ESP32_IP = "192.168.1.26"  # IP của S3 nãy ông gửi
ESP32_PORT = 82
SERVER_PORT = 8080         # Cổng của Laptop chạy Test
# ====================================================

# Hàm tự bắt IP Laptop
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        return s.getsockname()[0]
    except:
        return '127.0.0.1'
    finally:
        s.close()

LAPTOP_IP = get_local_ip()

# Dựng server FastAPI dùng chính thư mục hiện tại để host file test.mp3
app = FastAPI()
app.mount("/", StaticFiles(directory="."), name="static")

def trigger_esp32():
    time.sleep(2) # Đợi 2 giây cho Server Laptop khởi động xong
    
    # Đường dẫn file trên Laptop
    audio_url = f"http://{LAPTOP_IP}:{SERVER_PORT}/test1.mp3"
    # Lệnh bắn cho ESP32 (Đã thả xích volume=100 ở code C++)
    play_url = f"http://{ESP32_IP}:{ESP32_PORT}/play?url={audio_url}"
    
    print(f"\n[+] Đã mở Server cấp file tại: {audio_url}")
    print(f"🔥 BẮN LỆNH TỚI S3 KÉO MAX CÔNG SUẤT MAX98357A 🔥")
    
    try:
        response = requests.get(play_url, timeout=5)
        if response.status_code == 200:
            print("==================================================")
            print("🚀 ESP32 ĐÃ NHẬN LỆNH VÀ ĐANG KÉO FILE VỀ PHÁT!")
            print("❗ BẤM CTRL+C ĐỂ TẮT SERVER NẾU LOA CÓ MÙI KHÉT!")
            print("==================================================\n")
    except Exception as e:
        print(f"[!] Lỗi không gọi được mạch S3: {e}")

if __name__ == "__main__":
    if not os.path.exists("test.mp3"):
        print("[LỖI] Không tìm thấy file 'test.mp3' ở thư mục hiện tại!")
    else:
        # Chạy hàm bắn lệnh ngầm, song song đó khởi động server
        threading.Thread(target=trigger_esp32, daemon=True).start()
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")