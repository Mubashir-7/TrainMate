from pathlib import Path
from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .. import auth, models, database
from ..settings import settings
from ..flash import flash, get_flashed_messages
from ..hashing import Hash
import secrets

router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"],
    # The dependency is now injected into each route individually for clarity.
    # dependencies=[Depends(auth.get_current_user_for_page)],
)

templates = Jinja2Templates(directory="templates")

def _get_template_context(request: Request, user: models.User, **kwargs):
    """
    Helper to build the context dictionary for templates, ensuring
    common variables are always present.
    """
    # The user is now passed in explicitly.
    context = {
        "request": request,
        "current_user": user,
        "user": user, # for backwards compatibility with some templates
        "lora_active": settings.lora_path is not None and Path(settings.lora_path).exists(),
        "messages": get_flashed_messages(request),
    }
    context.update(kwargs)
    return context

@router.get("/", response_class=HTMLResponse, name="dashboard_home")
async def dashboard_home(request: Request, user: models.User = Depends(auth.get_current_user_for_page)):
    if isinstance(user, RedirectResponse):
        return user
    return templates.TemplateResponse("dashboard.html", _get_template_context(request, user=user))

@router.get("/chat", response_class=HTMLResponse, name="chat_history")
async def chat_history_page(request: Request, user: models.User = Depends(auth.get_current_user_for_page)):
    if isinstance(user, RedirectResponse):
        return user
    # Assuming history.html exists and works with the new context
    return templates.TemplateResponse("history.html", _get_template_context(request, user=user))

@router.get("/analytics", response_class=HTMLResponse, name="analytics")
async def analytics_page(request: Request, user: models.User = Depends(auth.get_current_user_for_page)):
    if isinstance(user, RedirectResponse):
        return user
    return templates.TemplateResponse("analytics.html", _get_template_context(request, user=user))

@router.get("/reports", response_class=HTMLResponse, name="reports")
async def reports_page(request: Request, user: models.User = Depends(auth.get_current_user_for_page)):
    if isinstance(user, RedirectResponse):
        return user
    return templates.TemplateResponse("reports.html", _get_template_context(request, user=user))

@router.get("/training", response_class=HTMLResponse, name="training")
async def training_page(request: Request, user: models.User = Depends(auth.get_current_user_for_page)):
    if isinstance(user, RedirectResponse):
        return user
    return templates.TemplateResponse("training.html", _get_template_context(request, user=user))

@router.get("/testchat", response_class=HTMLResponse, name="testchat")
async def testchat_page(request: Request, user: models.User = Depends(auth.get_current_user_for_page)):
    if isinstance(user, RedirectResponse):
        return user
    return templates.TemplateResponse("testchat.html", _get_template_context(request, user=user))

@router.get("/kommo", response_class=HTMLResponse, name="kommo")
async def kommo_page(
    request: Request,
    db: Session = Depends(database.get_db),
    user: models.User = Depends(auth.get_current_user_for_page)
):
    if isinstance(user, RedirectResponse):
        return user
    if not user.kommo_integration_key:
        user.kommo_integration_key = secrets.token_urlsafe(16)
        db.commit()
        db.refresh(user)
    return templates.TemplateResponse("kommo.html", _get_template_context(request, user=user))

@router.get("/settings", response_class=HTMLResponse, name="settings")
async def settings_page(
    request: Request, 
    db: Session = Depends(database.get_db),
    user: models.User = Depends(auth.get_current_user_for_page)
):
    if isinstance(user, RedirectResponse):
        return user
    # Ensure a profile object exists for the user before rendering the page.
    # This prevents an error if a user has never saved their API settings.
    if not user.profile:
        new_profile = models.UserProfile(user_id=user.id)
        db.add(new_profile)
        db.commit()
        db.refresh(user) # Refresh the user object to load the new profile
    return templates.TemplateResponse("settings.html", _get_template_context(request, user=user))

# --- Form Submission Endpoints for Settings Page ---

@router.post("/update-profile", name="update_profile")
async def update_profile(
    request: Request,
    db: Session = Depends(database.get_db),
    user: models.User = Depends(auth.get_current_user_for_page),
    username: str = Form(...),
    email: str = Form(...)
):
    if isinstance(user, RedirectResponse):
        return user
    # Check if the new username is already taken by another user
    if username != user.username and db.query(models.User).filter(models.User.username == username).first():
        flash(request, "Username already taken.", "error")
        return RedirectResponse(url=request.url_for("settings"), status_code=303)

    # Check if the new email is already taken by another user
    if email != user.email and db.query(models.User).filter(models.User.email == email).first():
        flash(request, "An account with this email already exists.", "error")
        return RedirectResponse(url=request.url_for("settings"), status_code=303)

    user.username = username
    user.email = email
    db.commit()

    # Log the user out for security reasons after changing credentials.
    # The login page will show the message from the query parameter.
    response = RedirectResponse(url="/login?message=Profile updated. Please log in again.", status_code=302)
    response.delete_cookie(key="access_token")
    return response

@router.post("/update-password", name="update_password")
async def update_password(
    request: Request,
    db: Session = Depends(database.get_db),
    user: models.User = Depends(auth.get_current_user_for_page),
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...)
):
    if isinstance(user, RedirectResponse):
        return user
    if not Hash.verify(user.hashed_password, current_password):
        flash(request, "Current password is not correct.", "error")
        return RedirectResponse(url=request.url_for("settings"), status_code=303)
    
    if new_password != confirm_password:
        flash(request, "New passwords do not match.", "error")
        return RedirectResponse(url=request.url_for("settings"), status_code=303)

    if len(new_password) < 8:
        flash(request, "New password must be at least 8 characters long.", "error")
        return RedirectResponse(url=request.url_for("settings"), status_code=303)

    user.hashed_password = Hash.bcrypt(new_password)
    db.commit()
    # Log the user out for security reasons after changing credentials.
    # The login page will show the message from the query parameter.
    response = RedirectResponse(url="/login?message=Password updated. Please log in again.", status_code=302)
    response.delete_cookie(key="access_token")
    return response

@router.post("/update-api-settings", name="update_api_settings")
async def update_api_settings(
    request: Request,
    db: Session = Depends(database.get_db),
    user: models.User = Depends(auth.get_current_user_for_page),
    xelence_affiliateid: str = Form(None),
    xelence_x_api_key: str = Form(None)
):
    if isinstance(user, RedirectResponse):
        return user
    profile = user.profile
    if not profile:
        profile = models.UserProfile(user_id=user.id)
        db.add(profile)

    profile.xelence_affiliateid = xelence_affiliateid
    if xelence_x_api_key and xelence_x_api_key != '••••••••':
        profile.xelence_x_api_key = xelence_x_api_key

    db.commit()
    flash(request, "API settings updated successfully!", "success")
    return RedirectResponse(url=request.url_for("settings"), status_code=303)