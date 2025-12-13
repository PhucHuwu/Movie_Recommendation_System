"""
Data cleaning and preprocessing script
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MIN_RATINGS_PER_USER, MIN_RATINGS_PER_MOVIE

def load_data():
    """Load raw data"""
    print("Loading raw data...")
    movies = pd.read_csv(RAW_DATA_DIR / "movies.csv")
    ratings = pd.read_csv(RAW_DATA_DIR / "ratings.csv")
    return movies, ratings

def clean_movies(movies):
    """Clean movies data"""
    print("\nCleaning movies data...")
    
    # Check for missing values
    print(f"Missing values before cleaning:\n{movies.isnull().sum()}")
    
    # Drop duplicates
    before = len(movies)
    movies = movies.drop_duplicates(subset=['movieId'])
    print(f"Removed {before - len(movies)} duplicate movies")
    
    # Handle missing values
    movies = movies.dropna(subset=['movieId', 'title'])
    movies['genres'] = movies['genres'].fillna('(no genres listed)')
    
    # Clean title (strip whitespace)
    movies['title'] = movies['title'].str.strip()
    
    # Parse genres into list
    movies['genres_list'] = movies['genres'].str.split('|')
    
    # Extract year from title (if present)
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$')
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    
    print(f"✓ Cleaned movies: {len(movies)} records")
    return movies

def clean_ratings(ratings, valid_movie_ids):
    """Clean ratings data"""
    print("\nCleaning ratings data...")
    
    # Check for missing values
    print(f"Missing values before cleaning:\n{ratings.isnull().sum()}")
    
    # Drop rows with missing values
    ratings = ratings.dropna()
    
    # Drop duplicates
    before = len(ratings)
    ratings = ratings.drop_duplicates(subset=['userId', 'movieId'])
    print(f"Removed {before - len(ratings)} duplicate ratings")
    
    # Remove outliers (ratings should be between 0.5 and 5.0)
    before = len(ratings)
    ratings = ratings[(ratings['rating'] >= 0.5) & (ratings['rating'] <= 5.0)]
    print(f"Removed {before - len(ratings)} outlier ratings")
    
    # Keep only ratings for movies that exist in movies dataset
    before = len(ratings)
    ratings = ratings[ratings['movieId'].isin(valid_movie_ids)]
    print(f"Removed {before - len(ratings)} ratings for non-existent movies")
    
    # Filter users and movies with minimum interactions
    print(f"\nFiltering by minimum interactions...")
    print(f"Min ratings per user: {MIN_RATINGS_PER_USER}")
    print(f"Min ratings per movie: {MIN_RATINGS_PER_MOVIE}")
    
    # Iteratively filter until convergence
    converged = False
    iteration = 0
    while not converged:
        iteration += 1
        prev_len = len(ratings)
        
        # Count ratings per user and movie
        user_counts = ratings['userId'].value_counts()
        movie_counts = ratings['movieId'].value_counts()
        
        # Filter
        valid_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
        valid_movies = movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index
        
        ratings = ratings[
            ratings['userId'].isin(valid_users) & 
            ratings['movieId'].isin(valid_movies)
        ]
        
        converged = (len(ratings) == prev_len)
        print(f"  Iteration {iteration}: {len(ratings)} ratings, {ratings['userId'].nunique()} users, {ratings['movieId'].nunique()} movies")
    
    print(f"✓ Cleaned ratings: {len(ratings)} records")
    return ratings

def save_cleaned_data(movies, ratings):
    """Save cleaned data"""
    print("\nSaving cleaned data...")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    movies.to_csv(PROCESSED_DATA_DIR / "movies_cleaned.csv", index=False)
    ratings.to_csv(PROCESSED_DATA_DIR / "ratings_cleaned.csv", index=False)
    
    print(f"✓ Saved to {PROCESSED_DATA_DIR}")

def print_statistics(movies, ratings):
    """Print dataset statistics"""
    print("\n" + "=" * 60)
    print("CLEANED DATA STATISTICS")
    print("=" * 60)
    
    print(f"\n📊 Movies:")
    print(f"   Total: {len(movies):,}")
    print(f"   Unique genres: {movies['genres'].nunique():,}")
    print(f"   Year range: {movies['year'].min():.0f} - {movies['year'].max():.0f}" if not movies['year'].isna().all() else "   Year range: N/A")
    
    print(f"\n📊 Ratings:")
    print(f"   Total: {len(ratings):,}")
    print(f"   Users: {ratings['userId'].nunique():,}")
    print(f"   Movies rated: {ratings['movieId'].nunique():,}")
    print(f"   Rating range: {ratings['rating'].min():.1f} - {ratings['rating'].max():.1f}")
    print(f"   Mean rating: {ratings['rating'].mean():.2f}")
    print(f"   Median rating: {ratings['rating'].median():.1f}")
    
    print(f"\n📊 Sparsity:")
    n_users = ratings['userId'].nunique()
    n_movies = ratings['movieId'].nunique()
    n_ratings = len(ratings)
    sparsity = 1 - (n_ratings / (n_users * n_movies))
    print(f"   Matrix size: {n_users:,} users × {n_movies:,} movies")
    print(f"   Sparsity: {sparsity:.2%}")
    
    print(f"\n📊 Top 5 genres:")
    genre_counts = movies['genres'].value_counts().head(5)
    for genre, count in genre_counts.items():
        print(f"   {genre}: {count}")

def main():
    """Main cleaning pipeline"""
    print("=" * 60)
    print("MOVIE RECOMMENDATION DATA CLEANING")
    print("=" * 60)
    
    # Load data
    movies, ratings = load_data()
    
    # Clean data
    movies = clean_movies(movies)
    ratings = clean_ratings(ratings, movies['movieId'].unique())
    
    # Print statistics
    print_statistics(movies, ratings)
    
    # Save cleaned data
    save_cleaned_data(movies, ratings)
    
    print("\n✓ Data cleaning completed successfully!")

if __name__ == "__main__":
    main()
