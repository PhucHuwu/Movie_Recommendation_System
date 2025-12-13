"""
Hybrid Recommendation Model
Combines User-Based CF, Item-Based CF, and Neural CF
"""
import numpy as np
import pickle

class HybridModel:
    def __init__(self, user_based_model, item_based_model, neural_cf_model,
                 weights=None):
        """
        Hybrid Recommendation Model
        
        Args:
            user_based_model: Trained UserBasedCF model
            item_based_model: Trained ItemBasedCF model
            neural_cf_model: Trained NeuralCF model
            weights: Dict with weights for each model (default: equal weights)
        """
        self.user_based = user_based_model
        self.item_based = item_based_model
        self.neural_cf = neural_cf_model
        
        if weights is None:
            self.weights = {
                'user_based': 1/3,
                'item_based': 1/3,
                'neural_cf': 1/3
            }
        else:
            # Normalize weights
            total = sum(weights.values())
            self.weights = {k: v/total for k, v in weights.items()}
        
        print(f"Hybrid model weights: {self.weights}")
    
    def predict(self, user_id, movie_id):
        """
        Predict rating using weighted combination
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        # Get predictions from all models
        pred_user = self.user_based.predict(user_id, movie_id)
        pred_item = self.item_based.predict(user_id, movie_id)
        pred_neural = self.neural_cf.predict(user_id, movie_id)
        
        # Weighted average
        prediction = (
            self.weights['user_based'] * pred_user +
            self.weights['item_based'] * pred_item +
            self.weights['neural_cf'] * pred_neural
        )
        
        # Clip to valid range
        return np.clip(prediction, 0.5, 5.0)
    
    def recommend(self, user_id, k=10, exclude_rated=True):
        """
        Generate top-k recommendations using weighted combination
        
        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        # Get recommendations from all models
        # Use larger k for individual models to have more candidates
        k_candidates = k * 3
        
        recs_user = self.user_based.recommend(user_id, k=k_candidates, exclude_rated=exclude_rated)
        recs_item = self.item_based.recommend(user_id, k=k_candidates, exclude_rated=exclude_rated)
        
        # For Neural CF, we need the set of rated movies
        if exclude_rated and user_id in self.user_based.user_id_map:
            user_idx = self.user_based.user_id_map[user_id]
            user_ratings = self.user_based.interaction_matrix[user_idx].toarray().flatten()
            rated_movie_indices = np.where(user_ratings > 0)[0]
            rated_movies = {self.user_based.idx_to_movie[idx] for idx in rated_movie_indices}
        else:
            rated_movies = set()
        
        recs_neural = self.neural_cf.recommend(
            user_id, 
            k=k_candidates, 
            exclude_rated=exclude_rated,
            rated_movies=rated_movies
        )
        
        # Collect all candidate movies
        all_movies = set()
        for movie_id, _ in recs_user + recs_item + recs_neural:
            all_movies.add(movie_id)
        
        # Compute hybrid score for each movie
        hybrid_scores = []
        for movie_id in all_movies:
            # Get prediction from combined model
            score = self.predict(user_id, movie_id)
            hybrid_scores.append((movie_id, score))
        
        # Sort by hybrid score
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_scores[:k]
    
    def save(self, filepath):
        """Save hybrid model configuration"""
        config = {
            'weights': self.weights
        }
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
        print(f"✓ Hybrid model config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, user_based_model, item_based_model, neural_cf_model):
        """Load hybrid model configuration"""
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        model = cls(
            user_based_model,
            item_based_model,
            neural_cf_model,
            weights=config['weights']
        )
        
        print(f"✓ Hybrid model loaded from {filepath}")
        return model
