"""
User-Based Collaborative Filtering Model - Optimized for Large Datasets
"""
import numpy as np
from scipy.sparse import csr_matrix
import pickle

class UserBasedCF:
    def __init__(self, k_neighbors=50, sample_users=None):
        """
        User-Based Collaborative Filtering
        
        Args:
            k_neighbors: Number of similar users to consider
            sample_users: Max users to use for similarity (None = all)
        """
        self.k_neighbors = k_neighbors
        self.sample_users = sample_users
        self.user_similarity = None
        self.interaction_matrix = None
        self.user_id_map = None
        self.movie_id_map = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.mean_rating = None
        self.sampled_user_indices = None
        
    def fit(self, interaction_matrix, user_id_map, movie_id_map, 
            idx_to_user, idx_to_movie, mean_rating):
        """
        Train the model by computing user-user similarity (sparse, top-k only)
        """
        print("Training User-Based CF...")
        
        self.interaction_matrix = interaction_matrix
        self.user_id_map = user_id_map
        self.movie_id_map = movie_id_map
        self.idx_to_user = idx_to_user
        self.idx_to_movie = idx_to_movie
        self.mean_rating = mean_rating
        
        n_users = interaction_matrix.shape[0]
        
        # Sample users if dataset is too large
        if self.sample_users and n_users > self.sample_users:
            print(f"  Sampling {self.sample_users:,} users from {n_users:,}...")
            self.sampled_user_indices = np.random.choice(
                n_users, self.sample_users, replace=False
            )
            matrix_for_sim = interaction_matrix[self.sampled_user_indices]
        else:
            self.sampled_user_indices = None
            matrix_for_sim = interaction_matrix
        
        # Compute top-k user similarities (sparse)
        print("Computing top-k user similarities...")
        from models.sparse_utils import sparse_cosine_similarity_top_k
        
        self.user_similarity = sparse_cosine_similarity_top_k(
            matrix_for_sim, 
            k=self.k_neighbors,
            batch_size=500
        )
        
        print(f"User similarity computed: {self.user_similarity.shape}")
        
    def predict(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        if user_id not in self.user_id_map or movie_id not in self.movie_id_map:
            return self.mean_rating
        
        user_idx = self.user_id_map[user_id]
        movie_idx = self.movie_id_map[movie_id]
        
        # Handle sampled case
        if self.sampled_user_indices is not None:
            # Find if user is in sampled set
            sampled_pos = np.where(self.sampled_user_indices == user_idx)[0]
            if len(sampled_pos) == 0:
                return self.mean_rating
            user_sim_idx = sampled_pos[0]
        else:
            user_sim_idx = user_idx
        
        # Get user similarities
        user_sims = self.user_similarity[user_sim_idx].toarray().flatten()
        
        # Get all user ratings for this movie
        movie_ratings = self.interaction_matrix[:, movie_idx].toarray().flatten()
        
        # Find similar users who rated this movie
        if self.sampled_user_indices is not None:
            rated_mask = movie_ratings[self.sampled_user_indices] > 0
            similar_users = np.where(rated_mask)[0]
            if len(similar_users) == 0:
                return self.mean_rating
            sims = user_sims[similar_users]
            ratings = movie_ratings[self.sampled_user_indices[similar_users]]
        else:
            rated_mask = movie_ratings > 0
            similar_users = np.where(rated_mask)[0]
            if len(similar_users) == 0:
                return self.mean_rating
            sims = user_sims[similar_users]
            ratings = movie_ratings[similar_users]
        
        # Weighted average
        if sims.sum() == 0:
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
        
        # Sample unrated movies if too many
        if len(unrated_movies) > 1000:
            unrated_movies = np.random.choice(unrated_movies, 1000, replace=False)
        
        # Predict ratings
        predictions = []
        for movie_idx in unrated_movies:
            movie_id = self.idx_to_movie[movie_idx]
            pred = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:k]
    
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
            'sample_users': self.sample_users,
            'user_similarity': self.user_similarity,
            'interaction_matrix': self.interaction_matrix,
            'user_id_map': self.user_id_map,
            'movie_id_map': self.movie_id_map,
            'idx_to_user': self.idx_to_user,
            'idx_to_movie': self.idx_to_movie,
            'mean_rating': self.mean_rating,
            'sampled_user_indices': self.sampled_user_indices
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = cls(k_neighbors=data['k_neighbors'], sample_users=data.get('sample_users'))
        for k, v in data.items():
            if k not in ['k_neighbors', 'sample_users']:
                setattr(model, k, v)
        print(f"Model loaded from {filepath}")
        return model
