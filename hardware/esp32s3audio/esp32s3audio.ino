#include <WiFi.h>
#include <WiFiManager.h>
#include <WebServer.h>
#include "Audio.h"
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ─── Pinout I2S (Audio) ───────────────────────────────────
#define I2S_BCLK  4
#define I2S_LRC   5
#define I2S_DOUT  6

// ─── Pinout I2C (LCD) ─────────────────────────────────────
#define I2C_SDA   8
#define I2C_SCL   9

#define AUDIO_PORT        82
#define WIFI_CHECK_MS     5000 

Audio audio;
WebServer audioServer(AUDIO_PORT);
LiquidCrystal_I2C lcd(0x27, 16, 2); 

static bool isPlaying = false;
static uint32_t lastWifiChk = 0;

void audio_info(const char* info) { Serial.printf("[Audio] %s\n", info); }
void audio_eof_mp3(const char* info) { Serial.printf("[EOF]   %s\n", info); isPlaying = false; }
void audio_error_mp3(const char* info) { Serial.printf("[ERR]   %s\n", info); isPlaying = false; }

void handleDisplay() {
  if (!audioServer.hasArg("text")) {
    audioServer.send(400, "text/plain", "Missing text");
    return;
  }
  String msg = audioServer.arg("text");
  Serial.println("[LCD] " + msg);
  lcd.clear();
  
  // [FIX 2]: Xử lý chuỗi cực kỳ an toàn, không sợ crash reset
  if (msg.length() <= 16) {
    lcd.setCursor(0, 0);
    lcd.print(msg);
  } else {
    lcd.setCursor(0, 0);
    lcd.print(msg.substring(0, 16));
    lcd.setCursor(0, 1);
    lcd.print(msg.substring(16)); // Bỏ số 32 đi
  }
  audioServer.send(200, "text/plain", "Displayed");
}

void handlePlay() {
  if (!audioServer.hasArg("url")) {
    audioServer.send(400, "text/plain", "Missing: url");
    return;
  }
  if (audioServer.hasArg("vol")) {
    int vol = audioServer.arg("vol").toInt();
    if (vol >= 0 && vol <= 100) audio.setVolume(vol);
  }
  const String& url = audioServer.arg("url");
  Serial.printf("[Play]  %s\n", url.c_str());

  audio.stopSong();
  isPlaying = audio.connecttohost(url.c_str());
  audioServer.send(isPlaying ? 200 : 502, "text/plain", isPlaying ? "Playing" : "Connect failed");
}

void handleStop() {
  audio.stopSong();
  isPlaying = false;
  audioServer.send(200, "text/plain", "Stopped");
}

// =========================================================
//  Task Audio: CHẠY TRÊN CORE 0 (Tách biệt hoàn toàn)
// =========================================================
void audioTask(void* param) {
    for (;;) {
        audio.loop();
        // Giữ delay 2ms để Core 0 không bị nóng chip
        vTaskDelay(2 / portTICK_PERIOD_MS); 
    }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(I2C_SDA, I2C_SCL);
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("AI Sign Language");
  lcd.setCursor(0, 1);
  lcd.print("Connecting Wi-Fi");

  WiFiManager wm;
  wm.setConnectTimeout(30);
  if (!wm.autoConnect("ESP32_S3_Audio", "12345678")) {
    lcd.clear();
    lcd.print("WiFi Failed!");
    delay(3000);
    ESP.restart();
  }
  
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("IP: ");
  lcd.print(WiFi.localIP());
  lcd.setCursor(0, 1);
  lcd.print("Ready to Speak!");

  audio.setPinout(I2S_BCLK, I2S_LRC, I2S_DOUT);
  audio.setTone(0, 0, 0);
  audio.setVolumeSteps(100);
  audio.setVolume(100); 

  audioServer.on("/play",    HTTP_GET, handlePlay);
  audioServer.on("/stop",    HTTP_GET, handleStop);
  audioServer.on("/display", HTTP_GET, handleDisplay);
  audioServer.begin();

  // [FIX 1]: Đuổi Task Audio sang Core 0, nâng độ ưu tiên lên Max (configMAX_PRIORITIES - 1)
  // Các thông số: Tên task, stack size, parameter, priority, task handle, Core ID
  xTaskCreatePinnedToCore(audioTask, "audioTask", 10000, NULL, configMAX_PRIORITIES - 1, NULL, 0);
  
  Serial.println("[System] Ready on Core 1! Audio is running on Core 0!");
}

void loop() {
    audioServer.handleClient(); // Xử lý web nằm gọn trên Core 1
    uint32_t now = millis();
    if (now - lastWifiChk >= WIFI_CHECK_MS) {
        lastWifiChk = now;
        if (WiFi.status() != WL_CONNECTED) {
            WiFi.reconnect();
        }
    }
}