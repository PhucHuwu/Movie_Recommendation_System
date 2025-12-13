"""
Script to download movie recommendation dataset from Kaggle
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR

def download_direct():
    """Download dataset directly using requests"""
    import requests
    import zipfile
    import tempfile
    
    print("Attempting direct download via requests...")
    
    # URLs to try - MovieLens is a great alternative with same structure
    urls = [
        ("Kaggle", "https://www.kaggle.com/api/v1/datasets/download/parasharmanas/movie-recommendation-system"),
        ("MovieLens", "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
    ]
    
    for name, url in urls:
        try:
            print(f"  Trying {name}: {url[:60]}...")
            
            response = requests.get(url, stream=True, timeout=120)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                print(f"  Downloading... ({total_size // 1024 // 1024} MB)")
                
                # Save to temp file
                temp_dir = tempfile.mkdtemp()
                zip_path = Path(temp_dir) / "dataset.zip"
                
                downloaded = 0
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                
                print(f"  Downloaded {downloaded // 1024} KB")
                
                # Extract zip
                print("  Extracting...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                return Path(temp_dir), name
            else:
                print(f"  HTTP {response.status_code} - trying next...")
                
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    return None, None

def copy_dataset_files(source_path, source_name):
    """Copy dataset files from source to RAW_DATA_DIR"""
    import shutil
    import pandas as pd
    
    # Find CSV files
    movies_file = None
    ratings_file = None
    
    for file in source_path.rglob("*.csv"):
        fname = file.name.lower()
        if "movies" in fname:
            movies_file = file
        elif "ratings" in fname:
            ratings_file = file
    
    if movies_file and ratings_file:
        # Copy to raw data directory
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(movies_file, RAW_DATA_DIR / "movies.csv")
        shutil.copy(ratings_file, RAW_DATA_DIR / "ratings.csv")
        
        print(f"\nCopied movies.csv to {RAW_DATA_DIR / 'movies.csv'}")
        print(f"Copied ratings.csv to {RAW_DATA_DIR / 'ratings.csv'}")
        
        # Validate dataset
        movies = pd.read_csv(RAW_DATA_DIR / "movies.csv")
        ratings = pd.read_csv(RAW_DATA_DIR / "ratings.csv")
        
        print(f"\nDataset Statistics ({source_name}):")
        print(f"   - Movies: {len(movies):,} records")
        print(f"   - Ratings: {len(ratings):,} records")
        print(f"   - Users: {ratings['userId'].nunique():,}")
        
        if len(movies) >= 2000:
            print(f"\nDataset meets minimum requirement of 2,000 movies")
        else:
            print(f"\nNote: Dataset has {len(movies)} movies")
        
        return True
    else:
        print(f"\nError: Could not find movies.csv or ratings.csv")
        print(f"  Found files: {list(source_path.rglob('*.csv'))}")
        return False

def download_dataset():
    """Download dataset"""
    print("=" * 60)
    print("Downloading Movie Recommendation Dataset...")
    print("=" * 60)
    
    # Download
    download_path, source_name = download_direct()
    
    if download_path and download_path.exists():
        print(f"\nDownloaded from {source_name}")
        success = copy_dataset_files(download_path, source_name)
        
        if success:
            print(f"\nDataset download completed successfully!")
            return True
    
    # Failed
    print("\n" + "=" * 60)
    print("Download failed.")
    print("=" * 60)
    print(f"\nPlease manually download and place files in: {RAW_DATA_DIR}")
    
    return False

if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)
