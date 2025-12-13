"""
Model training script
"""
import sys
from pathlib import Path
import pandas as pd
import pickle
from scipy.sparse import load_npz

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import FEATURES_DIR, SAVED_MODELS_DIR, NCF_EPOCHS, NCF_BATCH_SIZE, HYBRID_WEIGHTS
from models.user_based_cf import UserBasedCF
from models.item_based_cf import ItemBasedCF
from models.neural_cf import NeuralCF
from models.hybrid_model import HybridModel

def load_features():
    """Load preprocessed features"""
    print("Loading features...")
    
    # Load interaction matrix
    interaction_matrix = load_npz(FEATURES_DIR / "interaction_matrix.npz")
    
    # Load mappings
    with open(FEATURES_DIR / "mappings.pkl", 'rb') as f:
        mappings = pickle.load(f)
    
    # Load train/test data
    train_ratings = pd.read_csv(FEATURES_DIR / "train_ratings.csv")
    test_ratings = pd.read_csv(FEATURES_DIR / "test_ratings.csv")
    
    print("Features loaded")
    return interaction_matrix, mappings, train_ratings, test_ratings

def train_user_based_cf(interaction_matrix, mappings):
    """Train User-Based CF model"""
    print("\n" + "=" * 60)
    print("TRAINING USER-BASED CF")
    print("=" * 60)
    
    model = UserBasedCF(k_neighbors=50)
    model.fit(
        interaction_matrix,
        mappings['user_id_map'],
        mappings['movie_id_map'],
        mappings['idx_to_user'],
        mappings['idx_to_movie'],
        mappings['mean_rating']
    )
    
    # Save model
    model.save(SAVED_MODELS_DIR / "user_based_cf.pkl")
    
    return model

def train_item_based_cf(interaction_matrix, mappings):
    """Train Item-Based CF model"""
    print("\n" + "=" * 60)
    print("TRAINING ITEM-BASED CF")
    print("=" * 60)
    
    model = ItemBasedCF(k_neighbors=50)
    model.fit(
        interaction_matrix,
        mappings['user_id_map'],
        mappings['movie_id_map'],
        mappings['idx_to_user'],
        mappings['idx_to_movie'],
        mappings['mean_rating']
    )
    
    # Save model
    model.save(SAVED_MODELS_DIR / "item_based_cf.pkl")
    
    return model

def train_neural_cf(train_ratings, mappings):
    """Train Neural CF model"""
    print("\n" + "=" * 60)
    print("TRAINING NEURAL CF")
    print("=" * 60)
    
    model = NeuralCF(
        embedding_dim=32,
        hidden_layers=[64, 32, 16, 8],
        learning_rate=0.001
    )
    
    history = model.fit(
        train_ratings,
        mappings['user_id_map'],
        mappings['movie_id_map'],
        mappings['idx_to_user'],
        mappings['idx_to_movie'],
        mappings['mean_rating'],
        epochs=NCF_EPOCHS,
        batch_size=NCF_BATCH_SIZE,
        validation_split=0.1
    )
    
    # Save model
    model.save(SAVED_MODELS_DIR / "neural_cf.pkl")
    
    return model, history

def train_hybrid_model(user_based, item_based, neural_cf):
    """Create and save hybrid model"""
    print("\n" + "=" * 60)
    print("CREATING HYBRID MODEL")
    print("=" * 60)
    
    model = HybridModel(
        user_based,
        item_based,
        neural_cf,
        weights=HYBRID_WEIGHTS
    )
    
    # Save model configuration
    model.save(SAVED_MODELS_DIR / "hybrid_model.pkl")
    
    return model

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Create save directory
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load features
    interaction_matrix, mappings, train_ratings, test_ratings = load_features()
    
    # Train models
    user_based = train_user_based_cf(interaction_matrix, mappings)
    item_based = train_item_based_cf(interaction_matrix, mappings)
    neural_cf, history = train_neural_cf(train_ratings, mappings)
    hybrid = train_hybrid_model(user_based, item_based, neural_cf)
    
    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nModels saved to: {SAVED_MODELS_DIR}")
    print("\nNext steps:")
    print("  1. Run evaluation.py to evaluate models")
    print("  2. Start the backend server")
    print("  3. Launch the frontend")

if __name__ == "__main__":
    main()
