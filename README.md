# Driver Drowsiness Detection System using YOLOv8

Hệ thống phát hiện tài xế ngủ gật thời gian thực (Real-time) sử dụng thuật toán Deep Learning YOLOv8. Hệ thống có khả năng nhận diện trạng thái mắt và miệng để đưa ra cảnh báo tức thời, giúp giảm thiểu tai nạn giao thông.

##  Tính năng nổi bật
- **Phát hiện đa trạng thái:** Nhận diện mắt nhắm (Closed), mắt mở (Open) và hành động ngáp (Yawning).
- **Xử lý thời gian thực:** Đạt tốc độ **50+ FPS** trên GPU NVIDIA GTX.
- **Logic cảnh báo thông minh:** Phân biệt chớp mắt tự nhiên và ngủ quên (Microsleep).
- **Độ chính xác cao:** Huấn luyện trên bộ dữ liệu quy mô lớn (28,600+ ảnh).

##  Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.10
- **Deep Learning:** YOLOv8 (Ultralytics)
- **Thị giác máy tính:** OpenCV
- **Tăng tốc phần cứng:** NVIDIA CUDA & cuDNN

##  Cấu trúc thư mục
- `models/`: Chứa file trọng số mô hình `best.pt`.
- `train_yolo.py`: Script huấn luyện mô hình.
- `detect_yolo.py`: Script nhận diện thời gian thực qua Webcam.
- `data.yaml`: Cấu hình đường dẫn dataset.
- `requirements.txt`: Danh sách thư viện cần thiết.

##  Hướng dẫn cài đặt và sử dụng

1. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
2. **Chạy hệ thống nhận diện:**
   ```bash
   python detect_yolo.py   

##  Kết quả huấn luyện
- `tmAP@0.5`: ~95%
- `Precision`: ~93%
- `Recall`: ~91%
- `Inference Speed`: ~15ms/frame (trên GTX GPU)

##  Dataset
- Dự án sử dụng bộ dữ liệu Driver Drowsiness Detection từ Kaggle (Nexuswho).

##  Tác giả
- `Họ và tên`: Nguyễn Trí Trung.
- `Đề tài`: Đồ án môn học Deep Learning.

