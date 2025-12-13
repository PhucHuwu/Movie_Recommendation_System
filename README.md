# Movie Recommendation System

A machine learning-based movie recommendation system with multiple collaborative filtering models.

## Features

-   4 Recommendation Models: User-Based CF, Item-Based CF, Neural CF, Hybrid
-   Web Interface: Streamlit frontend with user and admin dashboards
-   REST API: FastAPI backend with MongoDB storage
-   Data Visualization: Rating distributions, genre analysis, model comparison

## Requirements

-   Python 3.10.16
-   MongoDB (for database features)
-   4GB+ RAM recommended (large dataset with 25M ratings)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/PhucHuwu/Movie_Recommendation_System.git
cd Movie_Recommendation_System
```

### 2. Create Conda Environment

```bash
conda create -n RCMsys python=3.10.16
conda activate RCMsys
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

```bash
python scripts/download_dataset.py
```

This downloads the MovieLens dataset from Kaggle (approx. 25M ratings, 60K movies).

### 5. Clean Data

```bash
python scripts/clean_data.py
```

### 6. Run Feature Engineering

```bash
python scripts/feature_engineering.py
```

### 7. Train Models

```bash
python scripts/train_models.py
```

Training takes 5-15 minutes depending on hardware.

### 8. Evaluate Models

```bash
python scripts/evaluation.py
```

### 9. Seed MongoDB

If using MongoDB for search and statistics:

```bash
# Start MongoDB first, then:
python scripts/seed_database.py
```

## Running the Application

### Start Backend Server

```bash
cd backend
python server.py
```

Backend runs at: http://localhost:8000

API Documentation: http://localhost:8000/docs

### Start Frontend (New Terminal)

```bash
cd frontend
streamlit run app.py
```

Frontend runs at: http://localhost:8501

## Project Structure

```
Movie_Recommendation_System/
├── backend/           # FastAPI server
├── frontend/          # Streamlit web app
├── models/            # ML model implementations
├── scripts/           # Data processing and training
├── data/              # Raw and processed data
└── config.py          # Configuration settings
```

## Models

| Model         | Description                        |
| ------------- | ---------------------------------- |
| User-Based CF | Recommends based on similar users  |
| Item-Based CF | Recommends based on similar items  |
| Neural CF     | MLP-based collaborative filtering  |
| Hybrid        | Weighted combination of all models |

## Evaluation Metrics

-   RMSE: Root Mean Square Error
-   MAE: Mean Absolute Error
-   Precision@K: Recommendation relevance
-   Recall@K: Coverage of relevant items

## Troubleshooting

**Import errors**: Ensure you are in the project root directory when running scripts.

**Memory errors**: The dataset is large. Try reducing sample sizes in config.py.

**MongoDB connection**: Make sure MongoDB is running before using database features.
