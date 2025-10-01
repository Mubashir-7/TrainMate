from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status, Request
from typing import Optional
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from sqlalchemy.orm import Session, joinedload
from .settings import settings
from . import models
from .database import get_db

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def verify_token(token: str):
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None

def get_current_user_for_page(request: Request, db: Session = Depends(get_db)):
    """
    A dependency for user-facing pages.
    Checks for a valid JWT token in the request cookies.
    If the token is not present or invalid, it redirects to the login page.
    If valid, it returns the corresponding user object from the database.
    """
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    username = verify_token(token)
    if username is None:
        response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
        response.delete_cookie(key="access_token")
        return response

    # Eagerly load the user's profile to ensure fresh data is always available.
    # This prevents issues where an updated API key might not be loaded.
    user = db.query(models.User).options(joinedload(models.User.profile)).filter(models.User.username == username).first()

    if user is None:
        response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
        response.delete_cookie(key="access_token")
        return response

    return user

def get_current_user_for_api(request: Request, db: Session = Depends(get_db)):
    """
    A dependency for API endpoints.
    Checks for a valid JWT token in the request cookies.
    If the token is not present or invalid, it raises an HTTPException.
    If valid, it returns the corresponding user object from the database.
    """
    token = request.cookies.get("access_token")
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    
    username = verify_token(token)
    if username is None:
        raise credentials_exception

    # Eagerly load the user's profile to ensure fresh data is always available.
    # This prevents issues where an updated API key might not be loaded.
    user = db.query(models.User).options(joinedload(models.User.profile)).filter(models.User.username == username).first()

    if user is None:
        raise credentials_exception

    return user