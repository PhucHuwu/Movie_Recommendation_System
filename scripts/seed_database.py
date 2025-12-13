"""
Helper script to seed MongoDB with data
"""
import sys
from pathlib import Path
import pandas as pd

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.database import db
from config import PROCESSED_DATA_DIR

def main():
    """Seed MongoDB with movies and ratings"""
    print("=" * 60)
    print("SEEDING DATABASE")
    print("=" * 60)
    
    # Load cleaned data
    print("\nLoading data...")
    movies = pd.read_csv(PROCESSED_DATA_DIR / "movies_cleaned.csv")
    ratings = pd.read_csv(PROCESSED_DATA_DIR / "ratings_cleaned.csv")
    
    # Seed database
    db.seed_data(movies, ratings)
    
    # Load and save evaluation metrics if available
    try:
        from config import SAVED_MODELS_DIR
        import json
        
        metrics_file = SAVED_MODELS_DIR / "evaluation_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                db.save_metrics(metrics)
    except:
        print("⚠ Evaluation metrics not found (run evaluation.py first)")
    
    print("\n✓ Database seeding completed!")
    print("\nYou can now start the backend server:")
    print("  cd backend")
    print("  python server.py")

if __name__ == "__main__":
    main()
