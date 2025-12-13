"""
Movies routes - Optimized with caching and sampling
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from database import db
from services.recommendation_service import recommendation_service
from functools import lru_cache
import time

router = APIRouter(prefix="/api/movies", tags=["movies"])

# Cache for top movies (computed once, cached for 10 minutes)
_top_movies_cache = {"data": None, "timestamp": 0}
_TOP_MOVIES_CACHE_TTL = 600  # 10 minutes

@router.get("/search")
async def search_movies(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=50),
    page: int = Query(1, ge=1)
):
    """
    Search movies by title or genre with pagination
    """
    skip = (page - 1) * limit
    
    # Use text index or regex search with limit
    query = {
        "$or": [
            {"title": {"$regex": q, "$options": "i"}},
            {"genres": {"$regex": q, "$options": "i"}}
        ]
    }
    
    total = db.movies.count_documents(query)
    results = list(db.movies.find(query).skip(skip).limit(limit))
    
    # Remove MongoDB _id field
    for movie in results:
        movie.pop('_id', None)
    
    return {
        "query": q,
        "count": len(results),
        "total": total,
        "page": page,
        "total_pages": (total + limit - 1) // limit,
        "results": results
    }

@router.get("/top/rated")
async def get_top_movies(limit: int = Query(20, ge=1, le=50)):
    """
    Get top rated movies (pre-computed from training data for speed)
    """
    global _top_movies_cache
    
    # Check cache
    current_time = time.time()
    if _top_movies_cache["data"] and (current_time - _top_movies_cache["timestamp"]) < _TOP_MOVIES_CACHE_TTL:
        return {"count": len(_top_movies_cache["data"][:limit]), "movies": _top_movies_cache["data"][:limit], "cached": True}
    
    # Compute top movies using sampled ratings (much faster)
    pipeline = [
        {"$sample": {"size": 100000}},  # Sample 100K ratings
        {"$group": {
            "_id": "$movieId",
            "avgRating": {"$avg": "$rating"},
            "numRatings": {"$sum": 1}
        }},
        {"$match": {"numRatings": {"$gte": 5}}},
        {"$sort": {"avgRating": -1}},
        {"$limit": 100}  # Cache top 100
    ]
    
    top_ratings = list(db.ratings.aggregate(pipeline))
    
    # Enrich with movie info
    top_movies = []
    for r in top_ratings:
        movie = db.movies.find_one({"movieId": r["_id"]})
        if movie:
            movie.pop('_id', None)
            movie['avgRating'] = r['avgRating']
            movie['numRatings'] = r['numRatings']
            top_movies.append(movie)
    
    # Update cache
    _top_movies_cache = {"data": top_movies, "timestamp": current_time}
    
    return {
        "count": len(top_movies[:limit]),
        "movies": top_movies[:limit],
        "cached": False
    }

@router.get("/genres/list")
async def get_genres():
    """
    Get list of all genres (cached via sampling)
    """
    # Sample movies for speed
    sample_movies = list(db.movies.aggregate([
        {"$sample": {"size": 5000}},
        {"$project": {"genres": 1}}
    ]))
    
    genres = set()
    for movie in sample_movies:
        genre_list = movie.get('genres', '').split('|')
        for genre in genre_list:
            genre = genre.strip()
            if genre and genre != '(no genres listed)':
                genres.add(genre)
    
    return {"genres": sorted(list(genres))}

@router.get("/{movie_id}")
async def get_movie(movie_id: int):
    """
    Get movie details by ID
    """
    movie = db.get_movie(movie_id)
    
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    movie.pop('_id', None)
    
    # Get similar movies (lazy loaded)
    try:
        similar = recommendation_service.get_similar_movies(movie_id, k=5)
        movie['similar_movies'] = similar
    except:
        movie['similar_movies'] = []
    
    return movie
