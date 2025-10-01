# app/main.py
from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from .settings import settings
from .routers import reports, auth_pages, dashboard_pages, chat_api, training_api, user_api #, kommo_api
from .model_manager import model_manager
from .training_status import training_status
from . import models
from .database import engine

models.Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI): # pylint: disable=W0613
    """Lifespan manager to pre-load the model at startup."""
    # Pre-load the chat model at startup. The full training model will be loaded
    # on-demand when a fine-tuning job is started. This makes startup much faster
    # and allows the app to run even if the large training model isn't downloaded.
    print("--- Pre-loading chat model at application startup ---")
    # model_manager.load_model_at_startup() # DEPRECATED: This tried to load the full training model.
    model_manager.get_chat_model() # This correctly loads only the chat model (GGUF by default).
    print("\n" + "="*80)
    print("✅✅✅ SERVER HAS STARTED/RELOADED! ✅✅✅")
    print("If you see this message, your code changes are being loaded correctly.")
    print("="*80 + "\n")
    print("🚀 Application startup complete. Chat model is loaded and ready.")
    yield
    print("🔌 Application shutdown.")

app = FastAPI(lifespan=lifespan)

# Add session middleware for flash messages
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)

# Include the router for our /api/reports endpoint
app.include_router(reports.router, prefix="/api", tags=["reports"])
app.include_router(auth_pages.router, tags=["Authentication"])
app.include_router(dashboard_pages.router, tags=["Dashboard"])
app.include_router(chat_api.router)
app.include_router(training_api.router)
app.include_router(user_api.router)
# app.include_router(kommo_api.router)

@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    """
    Middleware to add Cache-Control headers to prevent browser caching
    for all dashboard-related pages. This is crucial for ensuring
    authentication checks are always performed.
    """
    response = await call_next(request)
    # We need to ensure that both the HTML pages and the static assets they load
    # are not cached by the browser, especially during development.
    if request.url.path.startswith(("/dashboard", "/static", "/css")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Configure templates
templates = Jinja2Templates(directory="templates")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serves the application's favicon."""
    return FileResponse("templates/images/logo.svg")

# ----------------- 4) STATIC FILES AT ROOT -----------------
# Mount the CSS directory to serve files from `templates/css`
app.mount("/css", StaticFiles(directory="templates/css"), name="css")

# Mount the Images directory to serve files from `templates/images`
app.mount("/images", StaticFiles(directory="templates/images"), name="images")

# Mount the primary static directory for JS, etc., under a specific path
# to avoid conflicts with API routes like /dashboard.
app.mount("/static", StaticFiles(directory="static"), name="static_files")

# Add a specific route for the root path to serve the main chat widget's HTML.
# This ensures the chat widget is available at http://127.0.0.1:8000/
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the public homepage.
    """
    return templates.TemplateResponse("homepage.html", {"request": request})
