"""
Item-Based Collaborative Filtering Model - Optimized for Large Datasets
"""
import numpy as np
from scipy.sparse import csr_matrix
import pickle

class ItemBasedCF:
    def __init__(self, k_neighbors=50):
        """
        Item-Based Collaborative Filtering
        
        Args:
            k_neighbors: Number of similar items to consider
        """
        self.k_neighbors = k_neighbors
        self.item_similarity = None
        self.interaction_matrix = None
        self.user_id_map = None
        self.movie_id_map = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.mean_rating = None
        
    def fit(self, interaction_matrix, user_id_map, movie_id_map,
            idx_to_user, idx_to_movie, mean_rating):
        """
        Train by computing item-item similarity (sparse, top-k only)
        """
        print("Training Item-Based CF...")
        
        self.interaction_matrix = interaction_matrix
        self.user_id_map = user_id_map
        self.movie_id_map = movie_id_map
        self.idx_to_user = idx_to_user
        self.idx_to_movie = idx_to_movie
        self.mean_rating = mean_rating
        
        # Transpose for item-based (items x users)
        item_matrix = interaction_matrix.T
        
        print(f"  Item matrix shape: {item_matrix.shape}")
        
        # Compute top-k item similarities
        print("Computing top-k item similarities...")
        from models.sparse_utils import sparse_cosine_similarity_top_k
        
        self.item_similarity = sparse_cosine_similarity_top_k(
            item_matrix,
            k=self.k_neighbors,
            batch_size=500
        )
        
        print(f"✓ Item similarity computed: {self.item_similarity.shape}")
        
    def predict(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        if user_id not in self.user_id_map or movie_id not in self.movie_id_map:
            return self.mean_rating
        
        user_idx = self.user_id_map[user_id]
        movie_idx = self.movie_id_map[movie_id]
        
        # Get user's ratings
        user_ratings = self.interaction_matrix[user_idx].toarray().flatten()
        rated_movies = np.where(user_ratings > 0)[0]
        
        if len(rated_movies) == 0:
            return self.mean_rating
        
        # Get similarities between target movie and rated movies
        movie_sims = self.item_similarity[movie_idx].toarray().flatten()
        sims = movie_sims[rated_movies]
        ratings = user_ratings[rated_movies]
        
        # Weighted average
        if np.abs(sims).sum() < 1e-8:
            return self.mean_rating
        
        predicted = np.dot(sims, ratings) / (np.abs(sims).sum() + 1e-8)
        return np.clip(predicted, 0.5, 5.0)
    
    def recommend(self, user_id, k=10, exclude_rated=True):
        """Generate top-k recommendations"""
        if user_id not in self.user_id_map:
            return self._recommend_popular(k)
        
        user_idx = self.user_id_map[user_id]
        user_ratings = self.interaction_matrix[user_idx].toarray().flatten()
        
        if exclude_rated:
            unrated_movies = np.where(user_ratings == 0)[0]
        else:
            unrated_movies = np.arange(self.interaction_matrix.shape[1])
        
        # Sample if too many
        if len(unrated_movies) > 1000:
            unrated_movies = np.random.choice(unrated_movies, 1000, replace=False)
        
        predictions = []
        for movie_idx in unrated_movies:
            movie_id = self.idx_to_movie[movie_idx]
            pred = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:k]
    
    def get_similar_movies(self, movie_id, k=10):
        """Get similar movies"""
        if movie_id not in self.movie_id_map:
            return []
        
        movie_idx = self.movie_id_map[movie_id]
        sims = self.item_similarity[movie_idx].toarray().flatten()
        
        # Get top-k (excluding self)
        sims[movie_idx] = -1
        similar_idx = np.argsort(sims)[-(k):][::-1]
        
        return [(self.idx_to_movie[idx], sims[idx]) for idx in similar_idx if sims[idx] > 0]
    
    def _recommend_popular(self, k=10):
        """Recommend popular movies"""
        movie_ratings = []
        n_movies = min(1000, self.interaction_matrix.shape[1])
        for movie_idx in range(n_movies):
            ratings = self.interaction_matrix[:, movie_idx].toarray().flatten()
            ratings = ratings[ratings > 0]
            if len(ratings) > 0:
                movie_id = self.idx_to_movie[movie_idx]
                movie_ratings.append((movie_id, ratings.mean()))
        movie_ratings.sort(key=lambda x: x[1], reverse=True)
        return movie_ratings[:k]
    
    def save(self, filepath):
        """Save model"""
        model_data = {
            'k_neighbors': self.k_neighbors,
            'item_similarity': self.item_similarity,
            'interaction_matrix': self.interaction_matrix,
            'user_id_map': self.user_id_map,
            'movie_id_map': self.movie_id_map,
            'idx_to_user': self.idx_to_user,
            'idx_to_movie': self.idx_to_movie,
            'mean_rating': self.mean_rating
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = cls(k_neighbors=data['k_neighbors'])
        for k, v in data.items():
            if k != 'k_neighbors':
                setattr(model, k, v)
        print(f"✓ Model loaded from {filepath}")
        return model
