from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, auth, hashing
from ..database import get_db
from ..schemas import UserProfileUpdate

router = APIRouter(
    prefix="/api",
    tags=["User"]
)

@router.post("/settings", status_code=200)
async def update_user_settings(
    settings_data: UserProfileUpdate,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user_for_api)
):
    """
    Updates the user's profile information and password.
    """
    # Eagerly load the profile to ensure it's available
    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == user.id).one()

    # Dynamically update profile fields from the request data, excluding password fields.
    update_data = settings_data.model_dump(exclude_unset=True, exclude={'new_password', 'confirm_password'})
    for key, value in update_data.items():
        setattr(profile, key, value)
    
    # Handle password change
    if settings_data.new_password:
        if settings_data.new_password != settings_data.confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match.")
        if len(settings_data.new_password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters long.")
        
        user.hashed_password = hashing.Hash.bcrypt(settings_data.new_password)

    db.commit()
    return {"message": "Settings updated successfully."}