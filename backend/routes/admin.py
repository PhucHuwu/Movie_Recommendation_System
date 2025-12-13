"""
Admin routes for statistics and model evaluation
"""
from fastapi import APIRouter
from backend.database import db
from config import SAVED_MODELS_DIR
import json

router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.get("/statistics")
async def get_statistics():
    """
    Get dataset statistics
    """
    stats = db.get_statistics()
    return stats

@router.get("/visualizations/rating-distribution")
async def get_rating_distribution():
    """
    Get rating distribution data for visualization
    """
    pipeline = [
        {
            '$group': {
                '_id': '$rating',
                'count': {'$sum': 1}
            }
        },
        {
            '$sort': {'_id': 1}
        }
    ]
    
    results = list(db.ratings.aggregate(pipeline))
    
    return {
        "labels": [r['_id'] for r in results],
        "values": [r['count'] for r in results]
    }

@router.get("/visualizations/genre-distribution")
async def get_genre_distribution():
    """
    Get genre distribution data for visualization
    """
    all_movies = list(db.movies.find({}, {'genres': 1}))
    genre_counts = {}
    
    for movie in all_movies:
        genres = movie.get('genres', '').split('|')
        for genre in genres:
            genre = genre.strip()
            if genre and genre != '(no genres listed)':
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Sort by count
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "labels": [g[0] for g in sorted_genres[:15]],  # Top 15 genres
        "values": [g[1] for g in sorted_genres[:15]]
    }

@router.get("/visualizations/user-activity")
async def get_user_activity():
    """
    Get user activity distribution
    """
    pipeline = [
        {
            '$group': {
                '_id': '$userId',
                'count': {'$sum': 1}
            }
        },
        {
            '$bucket': {
                'groupBy': '$count',
                'boundaries': [1, 10, 50, 100, 500, 1000, 5000],
                'default': '5000+',
                'output': {
                    'users': {'$sum': 1}
                }
            }
        }
    ]
    
    results = list(db.ratings.aggregate(pipeline))
    
    return {
        "buckets": results
    }

@router.get("/models/metrics")
async def get_model_metrics():
    """
    Get evaluation metrics for all models
    """
    metrics_file = SAVED_MODELS_DIR / "evaluation_metrics.json"
    
    if not metrics_file.exists():
        return {
            "message": "Metrics not available. Please run evaluation.py first.",
            "metrics": []
        }
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return {
        "metrics": metrics
    }

@router.get("/models/comparison")
async def get_model_comparison():
    """
    Get model comparison data for visualization
    """
    metrics_file = SAVED_MODELS_DIR / "evaluation_metrics.json"
    
    if not metrics_file.exists():
        return {
            "message": "Metrics not available",
            "comparison": {}
        }
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Prepare comparison data
    models = [m['model'] for m in metrics]
    rmse = [m['rmse'] for m in metrics]
    mae = [m['mae'] for m in metrics]
    precision = [m['precision@10'] for m in metrics]
    recall = [m['recall@10'] for m in metrics]
    
    return {
        "models": models,
        "rmse": rmse,
        "mae": mae,
        "precision": precision,
        "recall": recall
    }
