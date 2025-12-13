"""
Recommendation service for loading and using models
Uses lazy loading to speed up server startup
"""
from pathlib import Path
import sys
import pickle

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import SAVED_MODELS_DIR, FEATURES_DIR
import pandas as pd

class RecommendationService:
    def __init__(self):
        """Initialize recommendation service with lazy loading"""
        self._models = {}
        self._movies_metadata = None
        self._loaded = False
        print("RecommendationService initialized (lazy loading enabled)")
    
    @property
    def models(self):
        """Lazy load models on first access"""
        if not self._loaded:
            self._load_models()
        return self._models
    
    @property
    def movies_metadata(self):
        """Lazy load metadata on first access"""
        if self._movies_metadata is None:
            self._load_metadata()
        return self._movies_metadata
    
    def _load_models(self):
        """Load all trained models"""
        if self._loaded:
            return
            
        print("Loading recommendation models (first request)...")
        
        try:
            from models.user_based_cf import UserBasedCF
            from models.item_based_cf import ItemBasedCF
            from models.neural_cf import NeuralCF
            from models.hybrid_model import HybridModel
            
            self._models['user_based'] = UserBasedCF.load(
                SAVED_MODELS_DIR / "user_based_cf.pkl"
            )
            self._models['item_based'] = ItemBasedCF.load(
                SAVED_MODELS_DIR / "item_based_cf.pkl"
            )
            self._models['neural_cf'] = NeuralCF.load(
                SAVED_MODELS_DIR / "neural_cf.pkl"
            )
            self._models['hybrid'] = HybridModel.load(
                SAVED_MODELS_DIR / "hybrid_model.pkl",
                self._models['user_based'],
                self._models['item_based'],
                self._models['neural_cf']
            )
            
            self._loaded = True
            print("All models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Make sure you have trained the models first!")
    
    def _load_metadata(self):
        """Load movie metadata"""
        try:
            self._movies_metadata = pd.read_csv(FEATURES_DIR / "movies_metadata.csv")
            print(f"Loaded metadata for {len(self._movies_metadata)} movies")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self._movies_metadata = pd.DataFrame()
    
    def is_ready(self):
        """Check if models are loaded"""
        return self._loaded
    
    def get_recommendations(self, user_id, model_name='hybrid', k=10):
        """
        Get recommendations for a user
        
        Args:
            user_id: User ID
            model_name: Model to use ('user_based', 'item_based', 'neural_cf', 'hybrid')
            k: Number of recommendations
            
        Returns:
            List of movie recommendations with metadata
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        # Get recommendations
        recommendations = model.recommend(user_id, k=k, exclude_rated=True)
        
        # Enrich with metadata
        enriched = []
        for movie_id, score in recommendations:
            movie_info = self.get_movie_info(movie_id)
            if movie_info:
                movie_info['predicted_rating'] = float(score)
                enriched.append(movie_info)
        
        return enriched
    
    def get_movie_info(self, movie_id):
        """Get movie metadata"""
        if self.movies_metadata is None or len(self.movies_metadata) == 0:
            return None
        
        movie = self.movies_metadata[self.movies_metadata['movieId'] == movie_id]
        
        if len(movie) == 0:
            return None
        
        movie = movie.iloc[0]
        return {
            'movieId': int(movie['movieId']),
            'title': movie['title'],
            'genres': movie['genres'],
            'year': int(movie['year']) if pd.notna(movie.get('year')) else None
        }
    
    def get_similar_movies(self, movie_id, k=10):
        """Get similar movies using item-based CF"""
        if 'item_based' not in self.models:
            return []
        
        similar = self.models['item_based'].get_similar_movies(movie_id, k=k)
        
        # Enrich with metadata
        enriched = []
        for mid, score in similar:
            movie_info = self.get_movie_info(mid)
            if movie_info:
                movie_info['similarity_score'] = float(score)
                enriched.append(movie_info)
        
        return enriched
    
    def predict_rating(self, user_id, movie_id, model_name='hybrid'):
        """Predict rating for a user-movie pair"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        prediction = model.predict(user_id, movie_id)
        
        return float(prediction)

# Global service instance (lazy loaded)
recommendation_service = RecommendationService()
