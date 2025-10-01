import secrets
from fastapi import APIRouter, Request, Depends, Form
from datetime import timedelta
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional
from fastapi.templating import Jinja2Templates

from .. import models, hashing, auth
from ..database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, message: Optional[str] = None):
    """Renders the login page."""
    return templates.TemplateResponse("login.html", {"request": request, "message": message})

@router.post("/login")
async def handle_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    remember_me: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    # The form uses 'username', so we query by username.
    user = db.query(models.User).filter(models.User.username == username).first()

    if not user or not hashing.Hash.verify(user.hashed_password, password):
        # On error, re-render the login page with an error message.
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password", "username": username})

    # The JWT 'sub' (subject) should still be the username for consistency with auth logic.
    expires_delta = None
    if remember_me:
        # Set a longer expiration for "Remember Me", e.g., 30 days
        expires_delta = timedelta(days=30)

    access_token = auth.create_access_token(data={"sub": user.username}, expires_delta=expires_delta)
    response = RedirectResponse(url="/dashboard", status_code=302)
    # Add headers to prevent the browser from caching this redirect
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

@router.get("/logout", status_code=302)
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie(key="access_token")
    return response

@router.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request, error: Optional[str] = None):
     """Renders the forgot password page."""
     return templates.TemplateResponse("forgot_password.html", {"request": request, "error": error})

@router.post("/forgot-password")
async def handle_forgot_password(request: Request, username: str = Form(...), db: Session = Depends(get_db)):
    """Handles the forgot password request."""
    #Basic implementation.  A real implementation would email a reset link.
    return templates.TemplateResponse("forgot_password.html", {"request": request, "message": "Password reset email sent (not really)."})

@router.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    """Renders the user signup page."""
    return templates.TemplateResponse("signup.html", {"request": request})

@router.post("/signup")
async def handle_signup(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Handles user registration."""
    if db.query(models.User).filter(models.User.username == username).first():
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Username already exists.", "username": username})

    if password != confirm_password:
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Passwords do not match.", "username": username})

    if len(password) < 8:
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Password must be at least 8 characters long.", "username": username})

    new_user = models.User(
        username=username,
        hashed_password=hashing.Hash.bcrypt(password),
        kommo_integration_key=secrets.token_urlsafe(9),
        profile=models.UserProfile()
    )
    db.add(new_user)
    db.commit()

    # Redirect to login page with a success message
    return RedirectResponse(url="/login?message=Signup successful! Please log in.", status_code=302)