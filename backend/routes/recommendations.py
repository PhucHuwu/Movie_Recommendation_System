"""
Recommendations routes
"""
from fastapi import APIRouter, HTTPException, Query
from services.recommendation_service import recommendation_service
from database import db

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])

@router.get("/{user_id}")
async def get_recommendations(
    user_id: int,
    model: str = Query("hybrid", description="Model to use: user_based, item_based, neural_cf, hybrid"),
    k: int = Query(10, ge=1, le=50, description="Number of recommendations")
):
    """
    Get personalized recommendations for a user
    """
    # Check if user exists
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Validate model name
    valid_models = ['user_based', 'item_based', 'neural_cf', 'hybrid']
    if model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Must be one of: {', '.join(valid_models)}"
        )
    
    try:
        recommendations = recommendation_service.get_recommendations(
            user_id=user_id,
            model_name=model,
            k=k
        )
        
        return {
            "userId": user_id,
            "model": model,
            "count": len(recommendations),
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/{user_id}/{movie_id}")
async def predict_rating(
    user_id: int,
    movie_id: int,
    model: str = Query("hybrid", description="Model to use")
):
    """
    Predict rating for a specific user-movie pair
    """
    try:
        prediction = recommendation_service.predict_rating(
            user_id=user_id,
            movie_id=movie_id,
            model_name=model
        )
        
        return {
            "userId": user_id,
            "movieId": movie_id,
            "model": model,
            "predictedRating": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}/ratings")
async def get_user_ratings(user_id: int):
    """
    Get user's rating history
    """
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    ratings = db.get_user_ratings(user_id)
    
    # Enrich with movie information
    enriched_ratings = []
    for rating in ratings:
        movie = db.get_movie(rating['movieId'])
        if movie:
            rating['movieTitle'] = movie['title']
            rating['movieGenres'] = movie['genres']
        rating.pop('_id', None)
        enriched_ratings.append(rating)
    
    return {
        "userId": user_id,
        "count": len(enriched_ratings),
        "ratings": enriched_ratings
    }
