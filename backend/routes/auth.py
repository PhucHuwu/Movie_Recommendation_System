"""
Authentication routes
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import db

router = APIRouter(prefix="/api/auth", tags=["auth"])

class LoginRequest(BaseModel):
    userId: int

class LoginResponse(BaseModel):
    userId: int
    message: str

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Login endpoint - validates that user exists in dataset
    """
    user = db.get_user(request.userId)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found in dataset")
    
    return LoginResponse(
        userId=request.userId,
        message="Login successful"
    )

@router.post("/logout")
async def logout():
    """
    Logout endpoint
    """
    return {"message": "Logout successful"}

@router.get("/me/{user_id}")
async def get_current_user(user_id: int):
    """
    Get current user information
    """
    user = db.get_user(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
# Get user's ratings
    ratings = db.get_user_ratings(user_id)
    
    return {
        "userId": user_id,
        "totalRatings": len(ratings),
        "avgRating": sum(r['rating'] for r in ratings) / len(ratings) if ratings else 0
    }
