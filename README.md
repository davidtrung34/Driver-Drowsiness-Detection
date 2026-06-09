Driver Drowsiness Detection System using YOLOv8
Hệ thống phát hiện tài xế ngủ gật thời gian thực (Real-time) sử dụng thuật toán Deep Learning YOLOv8. Hệ thống nhận diện trạng thái mắt và miệng để đưa ra cảnh báo tức thời giúp giảm thiểu tai nạn giao thông.
 Tính năng
Nhận diện các trạng thái: Mắt nhắm (Closed), Mắt mở (Open), Ngáp (Yawning).
Tốc độ xử lý: 50+ FPS (trên GPU NVIDIA GTX).
Logic cảnh báo: Phân biệt chớp mắt tự nhiên và ngủ quên (Microsleep).
Độ chính xác cao: Huấn luyện trên bộ dữ liệu 28,600 ảnh.
🛠 Công nghệ
Python 3.10
YOLOv8 (Ultralytics)
OpenCV
NVIDIA CUDA (Tăng tốc GPU)
 Cấu trúc thư mục
models/: Chứa file trọng số mô hình best.pt.
train_yolo.py: Code huấn luyện mô hình.
detect_yolo.py: Code nhận diện qua Webcam.
data.yaml: Cấu hình đường dẫn dữ liệu.
requirements.txt: Danh sách thư viện cần cài đặt.
 Hướng dẫn cài đặt và sử dụng
Cài đặt thư viện:
code
Bash
pip install -r requirements.txt
Chạy hệ thống nhận diện:
code
Bash
python detect_yolo.py
 Kết quả huấn luyện
mAP@0.5: ~95%
Tốc độ: ~15ms/khung hình.
Dữ liệu: Sử dụng Dataset Drowsiness Detection từ Kaggle.
 Tác giả
Họ và tên: Nguyễn Trí Trung.
Đồ án môn học: Deep Learning
