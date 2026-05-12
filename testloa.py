import socket
import threading
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

# --- CẤU HÌNH ---
ESP32_IP = "192.168.1.24"  # IP con S3 của ông
ESP32_PORT = 82
MP3_FILE = "test.mp3"      # Tên file MP3
SERVER_PORT = 8000         # Cổng của Laptop

# Hàm tự động dò IP của Laptop ông đang dùng
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

LAPTOP_IP = get_local_ip()

# Hàm mở Web Server ngầm trên Laptop
def start_server():
    httpd = TCPServer(("", SERVER_PORT), SimpleHTTPRequestHandler)
    print(f"[*] Python Web Server đã mở tại: http://{LAPTOP_IP}:{SERVER_PORT}")
    httpd.serve_forever()

print("--- CHƯƠNG TRÌNH TEST LOA ESP32 ---")

# 1. Bật Server chứa file MP3
t = threading.Thread(target=start_server, daemon=True)
t.start()
time.sleep(2) # Chờ 2s cho server ổn định

# 2. Tạo link cho ESP32 và bắn lệnh
audio_url = f"http://{LAPTOP_IP}:{SERVER_PORT}/{MP3_FILE}"
esp_command = f"http://{ESP32_IP}:{ESP32_PORT}/play?url={audio_url}"

print(f"[*] Đang yêu cầu ESP32 kéo file từ: {audio_url}")
try:
    req = urllib.request.urlopen(esp_command, timeout=5)
    print(f"[*] ESP32 trả lời: {req.read().decode('utf-8')}")
except Exception as e:
    print(f"[!] Lỗi kết nối tới ESP32: {e}")

print("\n[*] Hãy lắng nghe tiếng loa! Để im màn hình này cho ESP32 kéo file.")
print("[*] Bấm Ctrl+C để tắt...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[*] Đã tắt.")