"""
Feature engineering and train/test split
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, FEATURES_DIR, TRAIN_TEST_SPLIT, RANDOM_STATE

def load_cleaned_data():
    """Load cleaned data"""
    print("Loading cleaned data...")
    movies = pd.read_csv(PROCESSED_DATA_DIR / "movies_cleaned.csv")
    ratings = pd.read_csv(PROCESSED_DATA_DIR / "ratings_cleaned.csv")
    return movies, ratings

def create_genre_features(movies):
    """Create TF-IDF features from genres"""
    print("\nCreating genre features with TF-IDF...")
    
    # Replace pipe with space for tokenization
    genres_for_tfidf = movies['genres'].str.replace('|', ' ', regex=False)
    
    # TF-IDF vectorization of genres
    tfidf = TfidfVectorizer(
        lowercase=False,
        token_pattern=r'[A-Za-z-]+',  # Match genre names
    )
    
    genre_features = tfidf.fit_transform(genres_for_tfidf)
    
    print(f"Genre features shape: {genre_features.shape}")
    print(f"  Vocabulary size: {len(tfidf.vocabulary_)}")
    
    return genre_features, tfidf

def create_user_item_matrix(ratings, movies):
    """Create sparse user-item interaction matrix"""
    print("\nCreating user-item matrix...")
    
    # Create mappings
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    
    user_id_map = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
    movie_id_map = {mid: idx for idx, mid in enumerate(sorted(movie_ids))}
    
    # Inverse mappings
    idx_to_user = {idx: uid for uid, idx in user_id_map.items()}
    idx_to_movie = {idx: mid for mid, idx in movie_id_map.items()}
    
    # Map ratings to matrix indices
    ratings['user_idx'] = ratings['userId'].map(user_id_map)
    ratings['movie_idx'] = ratings['movieId'].map(movie_id_map)
    
    # Create sparse matrix
    n_users = len(user_id_map)
    n_movies = len(movie_id_map)
    
    interaction_matrix = csr_matrix(
        (ratings['rating'].values, 
         (ratings['user_idx'].values, ratings['movie_idx'].values)),
        shape=(n_users, n_movies)
    )
    
    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    print(f"  Users: {n_users:,}")
    print(f"  Movies: {n_movies:,}")
    print(f"  Density: {interaction_matrix.nnz / (n_users * n_movies):.4%}")
    
    return interaction_matrix, user_id_map, movie_id_map, idx_to_user, idx_to_movie

def stratified_user_split(ratings, test_size=0.2, random_state=42):
    """
    Split ratings by users to avoid domain shift.
    Each user's ratings are split into train/test.
    """
    print(f"\nPerforming stratified user split (test_size={test_size})...")
    
    train_list = []
    test_list = []
    
    # Group by user
    user_groups = ratings.groupby('userId')
    
    users_with_split = 0
    users_all_train = 0
    
    for user_id, user_ratings in user_groups:
        if len(user_ratings) >= 2:
            # Split this user's ratings
            user_train, user_test = train_test_split(
                user_ratings, 
                test_size=test_size,
                random_state=random_state
            )
            train_list.append(user_train)
            test_list.append(user_test)
            users_with_split += 1
        else:
            # User has only 1 rating, put in training
            train_list.append(user_ratings)
            users_all_train += 1
    
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    print(f"Split completed:")
    print(f"  Train: {len(train_df):,} ratings from {train_df['userId'].nunique():,} users")
    print(f"  Test: {len(test_df):,} ratings from {test_df['userId'].nunique():,} users")
    print(f"  Users with split: {users_with_split:,}")
    print(f"  Users all in train: {users_all_train:,}")
    
    return train_df, test_df

def normalize_ratings(ratings, train_ratings):
    """Normalize ratings using train set statistics"""
    print("\nNormalizing ratings...")
    
    mean_rating = train_ratings['rating'].mean()
    std_rating = train_ratings['rating'].std()
    
    ratings_copy = ratings.copy()
    ratings_copy['rating_normalized'] = (ratings['rating'] - mean_rating) / std_rating
    
    print(f"Normalization params: mean={mean_rating:.2f}, std={std_rating:.2f}")
    
    return ratings_copy, mean_rating, std_rating

def save_features(genre_features, tfidf, interaction_matrix, 
                  user_id_map, movie_id_map, idx_to_user, idx_to_movie,
                  train_ratings, test_ratings, movies,
                  mean_rating, std_rating):
    """Save all features and mappings"""
    print("\nSaving features...")
    
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save sparse matrices
    from scipy.sparse import save_npz
    save_npz(FEATURES_DIR / "genre_features.npz", genre_features)
    save_npz(FEATURES_DIR / "interaction_matrix.npz", interaction_matrix)
    
    # Save TF-IDF vectorizer
    with open(FEATURES_DIR / "tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(tfidf, f)
    
    # Save mappings
    mappings = {
        'user_id_map': user_id_map,
        'movie_id_map': movie_id_map,
        'idx_to_user': idx_to_user,
        'idx_to_movie': idx_to_movie,
        'mean_rating': mean_rating,
        'std_rating': std_rating
    }
    with open(FEATURES_DIR / "mappings.pkl", 'wb') as f:
        pickle.dump(mappings, f)
    
    # Save train/test splits
    train_ratings.to_csv(FEATURES_DIR / "train_ratings.csv", index=False)
    test_ratings.to_csv(FEATURES_DIR / "test_ratings.csv", index=False)
    
    # Save movie metadata (for recommendations)
    movies.to_csv(FEATURES_DIR / "movies_metadata.csv", index=False)
    
    print(f"Features saved to {FEATURES_DIR}")

def main():
    """Main feature engineering pipeline"""
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load cleaned data
    movies, ratings = load_cleaned_data()
    
    # Create genre features
    genre_features, tfidf = create_genre_features(movies)
    
    # Stratified user split
    train_ratings, test_ratings = stratified_user_split(
        ratings, 
        test_size=1-TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE
    )
    
    # Normalize ratings
    train_ratings, mean_rating, std_rating = normalize_ratings(train_ratings, train_ratings)
    test_ratings, _, _ = normalize_ratings(test_ratings, train_ratings)
    
    # Create user-item matrix (using only train data to avoid leakage)
    interaction_matrix, user_id_map, movie_id_map, idx_to_user, idx_to_movie = create_user_item_matrix(train_ratings, movies)
    
    # Save all features
    save_features(
        genre_features, tfidf, interaction_matrix,
        user_id_map, movie_id_map, idx_to_user, idx_to_movie,
        train_ratings, test_ratings, movies,
        mean_rating, std_rating
    )
    
    print("\nFeature engineering completed successfully!")

if __name__ == "__main__":
    main()
