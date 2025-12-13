"""
Neural Collaborative Filtering Model using sklearn MLPRegressor
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle

class NeuralCF:
    def __init__(self, hidden_layers=(64, 32, 16), learning_rate=0.001, max_iter=20):
        """
        Neural Collaborative Filtering using sklearn MLP
        
        Args:
            hidden_layers: Tuple of hidden layer sizes
            learning_rate: Learning rate for optimizer
            max_iter: Maximum training iterations
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.model = None
        self.user_id_map = None
        self.movie_id_map = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.mean_rating = None
        self.n_users = None
        self.n_movies = None
        
    def fit(self, train_ratings, user_id_map, movie_id_map, 
            idx_to_user, idx_to_movie, mean_rating,
            epochs=20, batch_size=None, validation_split=None):
        """
        Train the MLP model
        """
        print("Training Neural CF (sklearn MLP)...")
        
        self.user_id_map = user_id_map
        self.movie_id_map = movie_id_map
        self.idx_to_user = idx_to_user
        self.idx_to_movie = idx_to_movie
        self.mean_rating = mean_rating
        self.n_users = len(user_id_map)
        self.n_movies = len(movie_id_map)
        
        # Prepare data - sample if dataset is too large
        sample_size = min(500000, len(train_ratings))
        if len(train_ratings) > sample_size:
            print(f"  Sampling {sample_size:,} ratings for training...")
            train_sample = train_ratings.sample(n=sample_size, random_state=42)
        else:
            train_sample = train_ratings
        
        # Map user/movie IDs to indices
        train_sample = train_sample.copy()
        train_sample['user_idx'] = train_sample['userId'].map(user_id_map)
        train_sample['movie_idx'] = train_sample['movieId'].map(movie_id_map)
        
        # Remove any rows with unmapped values
        train_sample = train_sample.dropna(subset=['user_idx', 'movie_idx'])
        
        # Create features: user index and movie index
        X = train_sample[['user_idx', 'movie_idx']].values.astype(np.float32)
        # Normalize indices
        X[:, 0] = X[:, 0] / self.n_users
        X[:, 1] = X[:, 1] / self.n_movies
        
        y = train_sample['rating'].values
        
        print(f"  Training on {len(X):,} samples...")
        
        # Build and train model
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=True
        )
        
        self.model.fit(X, y)
        
        print("Neural CF training completed")
        return None
    
    def predict(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair
        """
        if user_id not in self.user_id_map or movie_id not in self.movie_id_map:
            return self.mean_rating
        
        user_idx = self.user_id_map[user_id]
        movie_idx = self.movie_id_map[movie_id]
        
        # Normalize
        X = np.array([[user_idx / self.n_users, movie_idx / self.n_movies]])
        
        prediction = self.model.predict(X)[0]
        
        # Clip to valid rating range
        return np.clip(prediction, 0.5, 5.0)
    
    def recommend(self, user_id, k=10, exclude_rated=True, rated_movies=None):
        """
        Generate top-k recommendations for a user
        """
        if user_id not in self.user_id_map:
            return self._recommend_popular(k)
        
        user_idx = self.user_id_map[user_id]
        
        # Get all movies
        all_movies = list(self.movie_id_map.keys())
        
        # Filter out rated movies if needed
        if exclude_rated and rated_movies is not None:
            candidate_movies = [m for m in all_movies if m not in rated_movies]
        else:
            candidate_movies = all_movies
        
        # Limit candidates for efficiency
        if len(candidate_movies) > 5000:
            import random
            candidate_movies = random.sample(candidate_movies, 5000)
        
        # Batch predict
        movie_indices = np.array([self.movie_id_map[m] for m in candidate_movies])
        X = np.column_stack([
            np.full(len(movie_indices), user_idx / self.n_users),
            movie_indices / self.n_movies
        ])
        
        predictions = self.model.predict(X)
        predictions = np.clip(predictions, 0.5, 5.0)
        
        # Create movie-rating pairs
        movie_ratings = list(zip(candidate_movies, predictions))
        movie_ratings.sort(key=lambda x: x[1], reverse=True)
        
        return movie_ratings[:k]
    
    def _recommend_popular(self, k=10):
        """Recommend popular movies for cold-start users"""
        import random
        movies = list(self.movie_id_map.keys())
        selected = random.sample(movies, min(k, len(movies)))
        return [(m, self.mean_rating) for m in selected]
    
    def save(self, filepath):
        """Save model to disk"""
        model_data = {
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'model': self.model,
            'user_id_map': self.user_id_map,
            'movie_id_map': self.movie_id_map,
            'idx_to_user': self.idx_to_user,
            'idx_to_movie': self.idx_to_movie,
            'mean_rating': self.mean_rating,
            'n_users': self.n_users,
            'n_movies': self.n_movies
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            hidden_layers=model_data['hidden_layers'],
            learning_rate=model_data['learning_rate'],
            max_iter=model_data['max_iter']
        )
        
        model.model = model_data['model']
        model.user_id_map = model_data['user_id_map']
        model.movie_id_map = model_data['movie_id_map']
        model.idx_to_user = model_data['idx_to_user']
        model.idx_to_movie = model_data['idx_to_movie']
        model.mean_rating = model_data['mean_rating']
        model.n_users = model_data['n_users']
        model.n_movies = model_data['n_movies']
        
        print(f"Model loaded from {filepath}")
        return model
