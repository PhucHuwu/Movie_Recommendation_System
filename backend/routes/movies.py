"""
Movies routes
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from backend.database import db
from backend.services.recommendation_service import recommendation_service

router = APIRouter(prefix="/api/movies", tags=["movies"])

@router.get("/search")
async def search_movies(
    q: str = Query(..., description="Search query"),
    limit: int = Query(50, ge=1, le=100)
):
    """
    Search movies by title or genre
    """
    results = db.search_movies(q, limit=limit)
    
    # Remove MongoDB _id field
    for movie in results:
        movie.pop('_id', None)
    
    return {
        "query": q,
        "count": len(results),
        "results": results
    }

@router.get("/{movie_id}")
async def get_movie(movie_id: int):
    """
    Get movie details by ID
    """
    movie = db.get_movie(movie_id)
    
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    movie.pop('_id', None)
    
    # Get similar movies
    similar = recommendation_service.get_similar_movies(movie_id, k=5)
    movie['similar_movies'] = similar
    
    return movie

@router.get("/top/rated")
async def get_top_movies(limit: int = Query(20, ge=1, le=100)):
    """
    Get top rated movies
    """
    # Get all movies with their ratings
    pipeline = [
        {
            '$lookup': {
                'from': 'ratings',
                'localField': 'movieId',
                'foreignField': 'movieId',
                'as': 'ratings'
            }
        },
        {
            '$addFields': {
                'avgRating': {'$avg': '$ratings.rating'},
                'numRatings': {'$size': '$ratings'}
            }
        },
        {
            '$match': {
                'numRatings': {'$gte': 10}  # Minimum 10 ratings
            }
        },
        {
            '$sort': {'avgRating': -1}
        },
        {
            '$limit': limit
        },
        {
            '$project': {
                '_id': 0,
                'ratings': 0
            }
        }
    ]
    
    top_movies = list(db.movies.aggregate(pipeline))
    
    return {
        "count": len(top_movies),
        "movies": top_movies
    }

@router.get("/genres/list")
async def get_genres():
    """
    Get list of all genres
    """
    all_movies = list(db.movies.find({}, {'genres': 1}))
    genres = set()
    
    for movie in all_movies:
        genre_list = movie.get('genres', '').split('|')
        for genre in genre_list:
            genre = genre.strip()
            if genre and genre != '(no genres listed)':
                genres.add(genre)
    
    return {
        "genres": sorted(list(genres))
    }
