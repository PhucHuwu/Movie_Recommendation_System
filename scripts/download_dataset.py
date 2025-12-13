"""
Script to download movie recommendation dataset from Kaggle
"""
import os
import sys
from pathlib import Path
import kagglehub

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR

def download_dataset():
    """Download dataset from Kaggle"""
    print("=" * 60)
    print("Downloading Movie Recommendation Dataset from Kaggle...")
    print("=" * 60)
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("parasharmanas/movie-recommendation-system")
        print(f"\n✓ Dataset downloaded to: {path}")
        
        # Copy files to raw data directory
        import shutil
        source_path = Path(path)
        
        # Find CSV files
        movies_file = None
        ratings_file = None
        
        for file in source_path.rglob("*.csv"):
            if "movies" in file.name.lower():
                movies_file = file
            elif "ratings" in file.name.lower():
                ratings_file = file
        
        if movies_file and ratings_file:
            # Copy to raw data directory
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(movies_file, RAW_DATA_DIR / "movies.csv")
            shutil.copy(ratings_file, RAW_DATA_DIR / "ratings.csv")
            
            print(f"\n✓ Copied movies.csv to {RAW_DATA_DIR / 'movies.csv'}")
            print(f"✓ Copied ratings.csv to {RAW_DATA_DIR / 'ratings.csv'}")
            
            # Validate dataset
            import pandas as pd
            movies = pd.read_csv(RAW_DATA_DIR / "movies.csv")
            ratings = pd.read_csv(RAW_DATA_DIR / "ratings.csv")
            
            print(f"\n📊 Dataset Statistics:")
            print(f"   - Movies: {len(movies):,} records")
            print(f"   - Ratings: {len(ratings):,} records")
            print(f"   - Users: {ratings['userId'].nunique():,}")
            
            if len(movies) >= 2000:
                print(f"\n✓ Dataset meets minimum requirement of 2,000 movies")
            else:
                print(f"\n⚠ Warning: Dataset has only {len(movies)} movies (minimum: 2,000)")
            
            print(f"\n✓ Dataset download completed successfully!")
            return True
            
        else:
            print("\n✗ Error: Could not find movies.csv or ratings.csv in downloaded dataset")
            return False
            
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nAlternative: You can manually download from:")
        print("https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system")
        print(f"Then place movies.csv and ratings.csv in: {RAW_DATA_DIR}")
        return False

if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)
