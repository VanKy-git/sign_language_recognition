# Hướng dẫn Workflow Cập nhật Dữ liệu & Huấn luyện Mô hình
Dự án: Nhận diện Ngôn ngữ Ký hiệu (Sign Language Recognition)

Quy trình dưới đây trình bày các bước chuẩn từ lúc thêm video quay thô cho đến khi chạy camera nhận diện thực tế. Hãy tuân thủ đúng thứ tự để tránh lỗi model.

## Bước 1: Chuẩn bị dữ liệu thô (Raw Videos)
1. Gom video quay được vào các folder, tên folder đặt theo tên từ vựng tiếng Anh (VD: `hello`, `thankyou`, `sorry`...).
2. Copy toàn bộ các folder từ vựng mới này bỏ vào thư mục `raw_videos/`.
   *Lưu ý: Hệ thống hiện tại đang lấy tối đa 50 video/từ (cấu hình `TARGET_QTY = 50`).*

## Bước 2: Trích xuất đặc trưng (Feature Extraction)
1. Mở terminal và chạy lệnh:
   ```bash
   python extract_data.py