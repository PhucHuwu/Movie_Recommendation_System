"""
Neural Collaborative Filtering Model
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

class NeuralCF:
    def __init__(self, embedding_dim=32, hidden_layers=[64, 32, 16, 8], 
                 learning_rate=0.001):
        """
        Neural Collaborative Filtering
        
        Args:
            embedding_dim: Dimension of user and item embeddings
            hidden_layers: List of hidden layer sizes
            learning_rate: Learning rate for optimizer
        """
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.model = None
        self.user_id_map = None
        self.movie_id_map = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.mean_rating = None
        self.n_users = None
        self.n_movies = None
        
    def build_model(self, n_users, n_movies):
        """
        Build the neural network architecture
        
        Args:
            n_users: Number of users
            n_movies: Number of movies
        """
        # User embedding path
        user_input = layers.Input(shape=(1,), name='user_input')
        user_embedding = layers.Embedding(
            n_users, 
            self.embedding_dim,
            name='user_embedding'
        )(user_input)
        user_vec = layers.Flatten(name='user_flatten')(user_embedding)
        
        # Movie embedding path
        movie_input = layers.Input(shape=(1,), name='movie_input')
        movie_embedding = layers.Embedding(
            n_movies,
            self.embedding_dim,
            name='movie_embedding'
        )(movie_input)
        movie_vec = layers.Flatten(name='movie_flatten')(movie_embedding)
        
        # Concatenate embeddings
        concat = layers.Concatenate(name='concat')([user_vec, movie_vec])
        
        # MLP layers
        x = concat
        for i, hidden_size in enumerate(self.hidden_layers):
            x = layers.Dense(
                hidden_size,
                activation='relu',
                name=f'hidden_{i}'
            )(x)
            x = layers.Dropout(0.2, name=f'dropout_{i}')(x)
        
        # Output layer
        output = layers.Dense(1, name='output')(x)
        
        # Build model
        model = keras.Model(
            inputs=[user_input, movie_input],
            outputs=output,
            name='NeuralCF'
        )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, train_ratings, user_id_map, movie_id_map, 
            idx_to_user, idx_to_movie, mean_rating,
            epochs=20, batch_size=256, validation_split=0.1):
        """
        Train the neural network
        
        Args:
            train_ratings: DataFrame with userId, movieId, rating columns
            user_id_map: Dict mapping userId to matrix index
            movie_id_map: Dict mapping movieId to matrix index
            idx_to_user: Dict mapping matrix index to userId
            idx_to_movie: Dict mapping matrix index to movieId
            mean_rating: Mean rating for normalization
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        print("Training Neural CF...")
        
        self.user_id_map = user_id_map
        self.movie_id_map = movie_id_map
        self.idx_to_user = idx_to_user
        self.idx_to_movie = idx_to_movie
        self.mean_rating = mean_rating
        self.n_users = len(user_id_map)
        self.n_movies = len(movie_id_map)
        
        # Build model
        self.model = self.build_model(self.n_users, self.n_movies)
        
        # Prepare training data
        train_ratings['user_idx'] = train_ratings['userId'].map(user_id_map)
        train_ratings['movie_idx'] = train_ratings['movieId'].map(movie_id_map)
        
        user_indices = train_ratings['user_idx'].values
        movie_indices = train_ratings['movie_idx'].values
        ratings = train_ratings['rating'].values
        
        # Train model
        history = self.model.fit(
            [user_indices, movie_indices],
            ratings,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        print("✓ Neural CF training completed")
        return history
    
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
        
        # Predict
        prediction = self.model.predict(
            [np.array([user_idx]), np.array([movie_idx])],
            verbose=0
        )[0][0]
        
        # Clip to valid rating range
        return np.clip(prediction, 0.5, 5.0)
    
    def recommend(self, user_id, k=10, exclude_rated=True, rated_movies=None):
        """
        Generate top-k recommendations for a user
        
        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_rated: Whether to exclude already rated movies
            rated_movies: Set of already rated movie IDs (for efficiency)
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_id_map:
            # New user - return popular movies
            return self._recommend_popular(k)
        
        user_idx = self.user_id_map[user_id]
        
        # Get all movies
        all_movies = list(self.movie_id_map.keys())
        
        # Filter out rated movies if needed
        if exclude_rated and rated_movies is not None:
            candidate_movies = [m for m in all_movies if m not in rated_movies]
        else:
            candidate_movies = all_movies
        
        # Batch predict for efficiency
        user_indices = np.array([user_idx] * len(candidate_movies))
        movie_indices = np.array([self.movie_id_map[m] for m in candidate_movies])
        
        predictions = self.model.predict(
            [user_indices, movie_indices],
            batch_size=512,
            verbose=0
        ).flatten()
        
        # Clip predictions
        predictions = np.clip(predictions, 0.5, 5.0)
        
        # Create movie-rating pairs
        movie_ratings = list(zip(candidate_movies, predictions))
        
        # Sort by predicted rating
        movie_ratings.sort(key=lambda x: x[1], reverse=True)
        
        return movie_ratings[:k]
    
    def _recommend_popular(self, k=10):
        """Recommend popular movies for cold-start users"""
        # Return random movies with mean rating (placeholder)
        import random
        movies = list(self.movie_id_map.keys())
        selected = random.sample(movies, min(k, len(movies)))
        return [(m, self.mean_rating) for m in selected]
    
    def save(self, filepath):
        """Save model to disk"""
        # Save Keras model
        model_path = filepath.replace('.pkl', '_keras.h5')
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'user_id_map': self.user_id_map,
            'movie_id_map': self.movie_id_map,
            'idx_to_user': self.idx_to_user,
            'idx_to_movie': self.idx_to_movie,
            'mean_rating': self.mean_rating,
            'n_users': self.n_users,
            'n_movies': self.n_movies
        }
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Model saved to {filepath} and {model_path}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk"""
        # Load metadata
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        model = cls(
            embedding_dim=metadata['embedding_dim'],
            hidden_layers=metadata['hidden_layers'],
            learning_rate=metadata['learning_rate']
        )
        
        # Load Keras model
        model_path = filepath.replace('.pkl', '_keras.h5')
        model.model = keras.models.load_model(model_path)
        
        # Restore metadata
        model.user_id_map = metadata['user_id_map']
        model.movie_id_map = metadata['movie_id_map']
        model.idx_to_user = metadata['idx_to_user']
        model.idx_to_movie = metadata['idx_to_movie']
        model.mean_rating = metadata['mean_rating']
        model.n_users = metadata['n_users']
        model.n_movies = metadata['n_movies']
        
        print(f"✓ Model loaded from {filepath}")
        return model
