import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# Model directories
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"

# Database configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB = os.getenv("MONGODB_DB", "movie_recommendation")

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Frontend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Model hyperparameters
DEFAULT_K = int(os.getenv("DEFAULT_K", 10))
MAX_K = int(os.getenv("MAX_K", 50))

# Training configuration
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
MIN_RATINGS_PER_USER = 5
MIN_RATINGS_PER_MOVIE = 10

# Neural CF hyperparameters
NCF_EMBEDDING_DIM = 32
NCF_HIDDEN_LAYERS = [64, 32, 16, 8]
NCF_EPOCHS = 20
NCF_BATCH_SIZE = 256
NCF_LEARNING_RATE = 0.001

# Hybrid model weights
HYBRID_WEIGHTS = {
    "user_based": 0.3,
    "item_based": 0.3,
    "neural_cf": 0.4
}

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, SAVED_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
