# Movie Recommendation System - Quick Start Guide

## Hướng dẫn triển khai nhanh

### Bước 1: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 2: Khởi động MongoDB

Đảm bảo MongoDB đang chạy. Nếu dùng MongoDB local:

```bash
mongod
```

Hoặc sử dụng MongoDB Atlas (cloud) và cập nhật connection string trong `.env`

### Bước 3: Download và chuẩn bị dữ liệu

```bash
# Download dataset từ Kaggle
python scripts/download_dataset.py

# Làm sạch dữ liệu
python scripts/data_cleaning.py

# Feature engineering
python scripts/feature_engineering.py
```

### Bước 4: Training models

```bash
# Train tất cả 4 models (có thể mất 10-30 phút)
python scripts/train_models.py

# Đánh giá models
python scripts/evaluation.py
```

### Bước 5: Seed database

```bash
python scripts/seed_database.py
```

### Bước 6: Khởi động Backend

```bash
cd backend
python server.py
```

Backend sẽ chạy tại: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

### Bước 7: Khởi động Frontend

Mở terminal mới:

```bash
cd frontend
streamlit run app.py
```

Frontend sẽ mở tại: `http://localhost:8501`

## Lưu ý

-   Dataset phải ≥ 2000 movies (đã đáp ứng)
-   Không có chức năng đăng ký user mới (bảo toàn dataset)
-   Admin interface có thể truy cập bởi tất cả users
-   System sử dụng 4 models: User-Based CF, Item-Based CF, Neural CF, Hybrid

## Troubleshooting

### Lỗi kết nối MongoDB

-   Kiểm tra MongoDB đang chạy
-   Kiểm tra connection string trong `.env`

### Lỗi import models

-   Chạy lại `python scripts/train_models.py`

### Frontend không kết nối được backend

-   Kiểm tra backend đang chạy ở port 8000
-   Kiểm tra BACKEND_URL trong `config.py`
