# Movie Recommendation System

Hệ thống gợi ý phim sử dụng Machine Learning với 4 mô hình: User-Based CF, Item-Based CF, Neural CF, và Hybrid Model.

## 📋 Tính năng chính

### User Interface

-   🔐 **Login**: Đăng nhập với userId từ dataset
-   🔍 **Search**: Tìm kiếm phim theo tên, thể loại
-   🎬 **Recommendations**: Gợi ý phim cá nhân hóa từ 4 mô hình AI
-   👤 **Profile**: Thông tin user và lịch sử đánh giá

### Admin Interface (Accessible by all users)

-   📊 **Statistics**: Thống kê dataset
-   📈 **Visualizations**: Trực quan hóa dữ liệu
-   🤖 **Model Evaluation**: So sánh hiệu suất các mô hình

## 🛠 Technology Stack

-   **Backend**: FastAPI + MongoDB
-   **Frontend**: Streamlit
-   **ML Models**: scikit-learn, TensorFlow
-   **Data Processing**: pandas, numpy, scipy

## 📂 Project Structure

```
Movie_Recommendation_System/
├── data/
│   ├── raw/              # Dataset gốc từ Kaggle
│   ├── processed/        # Dữ liệu đã làm sạch
│   └── features/         # Features đã vector hóa
├── models/
│   ├── saved/           # Model weights
│   ├── user_based_cf.py
│   ├── item_based_cf.py
│   ├── neural_cf.py
│   └── hybrid_model.py
├── scripts/
│   ├── download_dataset.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── evaluation.py
├── backend/
│   ├── server.py
│   ├── database.py
│   ├── routes/
│   └── services/
├── frontend/
│   ├── app.py
│   └── pages/
├── notebooks/
│   └── eda_notebook.ipynb
└── tests/
```

## 🚀 Installation

### 1. Clone repository

```bash
git clone <repo-url>
cd Movie_Recommendation_System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup environment

```bash
cp .env.example .env
# Edit .env với MongoDB connection string và Kaggle API key
```

### 4. Download dataset

```bash
python scripts/download_dataset.py
```

### 5. Prepare data và train models

```bash
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/train_models.py
```

## ▶️ Running the Application

### Start Backend

```bash
cd backend
uvicorn server:app --reload --port 8000
```

### Start Frontend

```bash
cd frontend
streamlit run app.py
```

Mở browser tại: `http://localhost:8501`

## 📊 Dataset

-   **Source**: [Kaggle - Movie Recommendation System](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)
-   **Files**:
    -   `movies.csv`: movieId, title, genres
    -   `ratings.csv`: userId, movieId, rating, timestamp
-   **Size**: ≥2000 movies

## 🤖 Models

1. **User-Based Collaborative Filtering**: Tìm users tương tự dựa trên rating patterns
2. **Item-Based Collaborative Filtering**: Tìm movies tương tự dựa trên user interactions
3. **Neural Collaborative Filtering**: Deep learning approach với embeddings
4. **Hybrid Model**: Kết hợp predictions từ 3 models trên

## 📈 Evaluation Metrics

-   RMSE (Root Mean Squared Error)
-   MAE (Mean Absolute Error)
-   Precision@K
-   Recall@K

## 🔒 Note

Hệ thống không cho phép users tạo rating mới để bảo toàn dataset gốc.

## 📄 License

MIT License
