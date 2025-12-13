"""
User-Based Collaborative Filtering Model
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class UserBasedCF:
    def __init__(self, k_neighbors=50):
        """
        User-Based Collaborative Filtering
        
        Args:
            k_neighbors: Number of similar users to consider
        """
        self.k_neighbors = k_neighbors
        self.user_similarity = None
        self.interaction_matrix = None
        self.user_id_map = None
        self.movie_id_map = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.mean_rating = None
        
    def fit(self, interaction_matrix, user_id_map, movie_id_map, 
            idx_to_user, idx_to_movie, mean_rating):
        """
        Train the model by computing user-user similarity
        
        Args:
            interaction_matrix: Sparse user-item matrix
            user_id_map: Dict mapping userId to matrix index
            movie_id_map: Dict mapping movieId to matrix index
            idx_to_user: Dict mapping matrix index to userId
            idx_to_movie: Dict mapping matrix index to movieId
            mean_rating: Mean rating for normalization
        """
        print("Training User-Based CF...")
        
        self.interaction_matrix = interaction_matrix
        self.user_id_map = user_id_map
        self.movie_id_map = movie_id_map
        self.idx_to_user = idx_to_user
        self.idx_to_movie = idx_to_movie
        self.mean_rating = mean_rating
        
        # Compute user-user similarity using cosine similarity
        print("Computing user-user similarity matrix...")
        self.user_similarity = cosine_similarity(interaction_matrix, dense_output=False)
        
        print(f"✓ User similarity matrix shape: {self.user_similarity.shape}")
        
    def predict(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_id_map or movie_id not in self.movie_id_map:
            return self.mean_rating
        
        user_idx = self.user_id_map[user_id]
        movie_idx = self.movie_id_map[movie_id]
        
        # Get user similarities
        user_sims = self.user_similarity[user_idx].toarray().flatten()
        
        # Get users who rated this movie
        movie_ratings = self.interaction_matrix[:, movie_idx].toarray().flatten()
        rated_users = np.where(movie_ratings > 0)[0]
        
        if len(rated_users) == 0:
            return self.mean_rating
        
        # Get similarities and ratings for users who rated this movie
        sims = user_sims[rated_users]
        ratings = movie_ratings[rated_users]
        
        # Get top-k similar users
        if len(sims) > self.k_neighbors:
            top_k_idx = np.argsort(sims)[-self.k_neighbors:]
            sims = sims[top_k_idx]
            ratings = ratings[top_k_idx]
        
        # Weighted average
        if sims.sum() == 0:
            return self.mean_rating
        
        predicted = np.dot(sims, ratings) / sims.sum()
        
        # Clip to valid rating range
        return np.clip(predicted, 0.5, 5.0)
    
    def recommend(self, user_id, k=10, exclude_rated=True):
        """
        Generate top-k recommendations for a user
        
        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_id_map:
            # New user - return popular movies
            return self._recommend_popular(k)
        
        user_idx = self.user_id_map[user_id]
        
        # Get movies the user hasn't rated
        user_ratings = self.interaction_matrix[user_idx].toarray().flatten()
        
        if exclude_rated:
            unrated_movies = np.where(user_ratings == 0)[0]
        else:
            unrated_movies = np.arange(self.interaction_matrix.shape[1])
        
        # Predict ratings for all unrated movies
        predictions = []
        for movie_idx in unrated_movies:
            movie_id = self.idx_to_movie[movie_idx]
            pred_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:k]
    
    def _recommend_popular(self, k=10):
        """Recommend popular movies for cold-start users"""
        # Calculate average rating for each movie
        movie_ratings = []
        for movie_idx in range(self.interaction_matrix.shape[1]):
            ratings = self.interaction_matrix[:, movie_idx].toarray().flatten()
            ratings = ratings[ratings > 0]
            if len(ratings) > 0:
                avg_rating = ratings.mean()
                movie_id = self.idx_to_movie[movie_idx]
                movie_ratings.append((movie_id, avg_rating))
        
        movie_ratings.sort(key=lambda x: x[1], reverse=True)
        return movie_ratings[:k]
    
    def save(self, filepath):
        """Save model to disk"""
        model_data = {
            'k_neighbors': self.k_neighbors,
            'user_similarity': self.user_similarity,
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
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(k_neighbors=model_data['k_neighbors'])
        model.user_similarity = model_data['user_similarity']
        model.interaction_matrix = model_data['interaction_matrix']
        model.user_id_map = model_data['user_id_map']
        model.movie_id_map = model_data['movie_id_map']
        model.idx_to_user = model_data['idx_to_user']
        model.idx_to_movie = model_data['idx_to_movie']
        model.mean_rating = model_data['mean_rating']
        
        print(f"✓ Model loaded from {filepath}")
        return model
