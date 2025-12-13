"""
Admin routes for statistics and model evaluation
Optimized with pagination and data sampling for large datasets
"""
from fastapi import APIRouter, Query
from database import db
from config import SAVED_MODELS_DIR
import json

router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.get("/statistics")
async def get_statistics():
    """
    Get dataset statistics (cached/sampled for speed)
    """
    # Use MongoDB aggregation with limits for speed
    stats = {
        "total_movies": db.movies.estimated_document_count(),
        "total_ratings": db.ratings.estimated_document_count(),
        "total_users": len(db.ratings.distinct("userId", {})),
    }
    
    # Sample for average rating (faster than full scan)
    pipeline = [
        {"$sample": {"size": 10000}},
        {"$group": {"_id": None, "avg": {"$avg": "$rating"}}}
    ]
    result = list(db.ratings.aggregate(pipeline))
    stats["avg_rating"] = round(result[0]["avg"], 2) if result else 0
    
    return stats

@router.get("/visualizations/rating-distribution")
async def get_rating_distribution():
    """
    Get rating distribution (sampled for speed)
    """
    # Sample 50K ratings for distribution
    pipeline = [
        {"$sample": {"size": 50000}},
        {"$group": {"_id": "$rating", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    
    results = list(db.ratings.aggregate(pipeline))
    
    return {
        "labels": [r['_id'] for r in results],
        "values": [r['count'] for r in results],
        "note": "Sampled from 50,000 ratings"
    }

@router.get("/visualizations/genre-distribution")
async def get_genre_distribution(limit: int = Query(15, ge=1, le=30)):
    """
    Get genre distribution (limited movies scan)
    """
    # Only scan limited movies for speed
    all_movies = list(db.movies.find({}, {'genres': 1}).limit(10000))
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
        "labels": [g[0] for g in sorted_genres[:limit]],
        "values": [g[1] for g in sorted_genres[:limit]]
    }

@router.get("/visualizations/user-activity")
async def get_user_activity():
    """
    Get user activity distribution (sampled)
    """
    # Sample users for speed
    pipeline = [
        {"$sample": {"size": 100000}},
        {"$group": {"_id": "$userId", "count": {"$sum": 1}}},
        {"$bucket": {
            "groupBy": "$count",
            "boundaries": [1, 10, 50, 100, 500, 1000, 5000],
            "default": "5000+",
            "output": {"users": {"$sum": 1}}
        }}
    ]
    
    results = list(db.ratings.aggregate(pipeline))
    
    return {"buckets": results}

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
    
    # Convert ObjectId to string
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

@router.get("/health")
async def health_check():
    """Quick health check endpoint"""
    return {"status": "ok", "message": "Admin API is running"}
