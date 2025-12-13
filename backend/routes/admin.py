"""
Admin routes for statistics and model evaluation
Full dataset loading with caching for visualizations
"""
from fastapi import APIRouter, Query
from database import db
from config import SAVED_MODELS_DIR
import json
import time

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Server-side cache for visualization data
_cache = {
    "statistics": {"data": None, "timestamp": 0},
    "rating_distribution": {"data": None, "timestamp": 0},
    "genre_distribution": {"data": None, "timestamp": 0},
}
_CACHE_TTL = 3600  # 1 hour cache

def get_cached(key):
    """Get from cache if not expired"""
    if _cache[key]["data"] and (time.time() - _cache[key]["timestamp"]) < _CACHE_TTL:
        return _cache[key]["data"]
    return None

def set_cached(key, data):
    """Set cache"""
    _cache[key] = {"data": data, "timestamp": time.time()}

@router.get("/statistics")
async def get_statistics():
    """
    Get dataset statistics (cached)
    """
    cached = get_cached("statistics")
    if cached:
        return {**cached, "cached": True}
    
    # Full count
    stats = {
        "total_movies": db.movies.count_documents({}),
        "total_ratings": db.ratings.count_documents({}),
        "total_users": len(db.ratings.distinct("userId")),
    }
    
    # Full average rating calculation
    pipeline = [
        {"$group": {"_id": None, "avg": {"$avg": "$rating"}}}
    ]
    result = list(db.ratings.aggregate(pipeline, allowDiskUse=True))
    stats["avg_rating"] = round(result[0]["avg"], 2) if result else 0
    
    set_cached("statistics", stats)
    return {**stats, "cached": False}

@router.get("/visualizations/rating-distribution")
async def get_rating_distribution():
    """
    Get rating distribution (FULL dataset, cached)
    """
    cached = get_cached("rating_distribution")
    if cached:
        return {**cached, "cached": True}
    
    # Full aggregation
    pipeline = [
        {"$group": {"_id": "$rating", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    
    results = list(db.ratings.aggregate(pipeline, allowDiskUse=True))
    
    data = {
        "labels": [r['_id'] for r in results],
        "values": [r['count'] for r in results]
    }
    
    set_cached("rating_distribution", data)
    return {**data, "cached": False}

@router.get("/visualizations/genre-distribution")
async def get_genre_distribution(limit: int = Query(15, ge=1, le=30)):
    """
    Get genre distribution (FULL dataset, cached)
    """
    cached = get_cached("genre_distribution")
    if cached:
        return {
            "labels": cached["labels"][:limit],
            "values": cached["values"][:limit],
            "cached": True
        }
    
    # Full movies scan
    all_movies = list(db.movies.find({}, {'genres': 1}))
    genre_counts = {}
    
    for movie in all_movies:
        genres = movie.get('genres', '').split('|')
        for genre in genres:
            genre = genre.strip()
            if genre and genre != '(no genres listed)':
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    
    data = {
        "labels": [g[0] for g in sorted_genres],
        "values": [g[1] for g in sorted_genres]
    }
    
    set_cached("genre_distribution", data)
    return {
        "labels": data["labels"][:limit],
        "values": data["values"][:limit],
        "cached": False
    }

@router.get("/movies/list")
async def get_movies_paginated(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    search: str = Query(None)
):
    """
    Get paginated movies list
    """
    query = {}
    if search:
        query["title"] = {"$regex": search, "$options": "i"}
    
    skip = (page - 1) * per_page
    
    movies = list(db.movies.find(query).skip(skip).limit(per_page))
    total = db.movies.count_documents(query)
    
    for m in movies:
        m['_id'] = str(m['_id'])
    
    return {
        "movies": movies,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page
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
    
    return {"metrics": metrics}

@router.get("/models/comparison")
async def get_model_comparison():
    """
    Get model comparison data for visualization
    """
    metrics_file = SAVED_MODELS_DIR / "evaluation_metrics.json"
    
    if not metrics_file.exists():
        return {"message": "Metrics not available", "comparison": {}}
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
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

@router.get("/cache/clear")
async def clear_cache():
    """Clear all visualization cache"""
    global _cache
    _cache = {
        "statistics": {"data": None, "timestamp": 0},
        "rating_distribution": {"data": None, "timestamp": 0},
        "genre_distribution": {"data": None, "timestamp": 0},
    }
    return {"message": "Cache cleared"}

@router.get("/health")
async def health_check():
    """Quick health check endpoint"""
    return {"status": "ok", "message": "Admin API is running"}
