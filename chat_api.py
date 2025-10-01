from pathlib import Path
import re
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, joinedload

from .. import models, auth
from ..database import get_db
from ..schemas import ChatRequest, SessionOut, MessageOut
from ..model_loader import generate_stream
from ..model_manager import model_manager
from ..settings import settings
from typing import List, Optional

router = APIRouter(
    prefix="/api",
    tags=["Chat"]
)

def remove_emojis(text: str) -> str:
    """Removes most emoji Unicode ranges."""
    # This regex covers most of the emoji characters.
    return re.sub(r'[\U00010000-\U0010ffff]', '', text)

def _get_or_create_session(db: Session, user: models.User, session_id: Optional[int], user_message_content: str) -> models.ChatSession:
    """
    A helper function to get an existing chat session or create a new one.
    This encapsulates the session management logic, making the main chat endpoint cleaner.
    """
    session = None
    if session_id:
        session = db.query(models.ChatSession).filter(
            models.ChatSession.id == session_id,
            models.ChatSession.user_id == user.id
        ).first()

    if not session:
        # Create a title from the first 75 chars of the user's message
        title = (user_message_content[:75] + '...') if len(user_message_content) > 75 else user_message_content
        session = models.ChatSession(user_id=user.id, title=title)
        db.add(session)
        db.commit()
        db.refresh(session)
    return session

def _save_message_to_db(db: Session, session_id: int, role: str, content: str):
    """Saves a chat message to the database."""
    message = models.ChatMessage(session_id=session_id, role=role, content=content)
    db.add(message)
    db.commit()

@router.post("/chat")
async def chat(
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user_for_api)
):
    user_message_content = chat_request.history[-1].content if chat_request.history else ""
    if not user_message_content:
        raise HTTPException(status_code=400, detail="Cannot process an empty message.")

    history_dicts = [msg.model_dump() for msg in chat_request.history]

    # --- Real-time Instruction Loading ---
    # On every chat request, load the bot's instructions from the file.
    # This ensures that any changes made on the dashboard are applied immediately.
    system_prompt = None
    try:
        instructions_path = Path("data/bot_instructions.txt")
        if instructions_path.exists() and instructions_path.stat().st_size > 0:
            system_prompt = instructions_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Warning: Could not load bot instructions at runtime: {e}")

    async def stream_and_save_logic():
        # 1. Get or create the session for this conversation
        session = _get_or_create_session(db, user, chat_request.session_id, user_message_content)

        # 2. Save the user's message to the database
        _save_message_to_db(db, session.id, "user", user_message_content)

        # 3. First, yield the session ID so the client can update its state
        yield f"session_id:{session.id}\n\n"

        # 4. Get the model and generate the response stream
        model, tokenizer = model_manager.get_chat_model()
        stream = generate_stream(model, tokenizer, history_dicts, settings.max_tokens, settings.temperature, system_prompt=system_prompt)

        assistant_response_content = ""
        for chunk in stream:
            # Clean the chunk to remove emojis before adding it to the full response and yielding it.
            clean_chunk = remove_emojis(chunk)
            assistant_response_content += clean_chunk
            yield clean_chunk

        # 5. After the stream is complete, save the full assistant response
        # The content is already clean because we cleaned each chunk.
        _save_message_to_db(db, session.id, "assistant", assistant_response_content)

    return StreamingResponse(stream_and_save_logic(), media_type="text/event-stream")

@router.get("/sessions", response_model=List[SessionOut])
async def get_sessions(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user_for_api)
):
    sessions = db.query(models.ChatSession).filter(models.ChatSession.user_id == user.id).order_by(models.ChatSession.created_at.desc()).all()
    return sessions

@router.get("/sessions/{session_id}", response_model=List[MessageOut])
async def get_session_messages(
    session_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user_for_api)
):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.messages

@router.delete("/sessions/all", status_code=200)
async def delete_all_sessions(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user_for_api)
):
    try:
        sessions = db.query(models.ChatSession).filter(models.ChatSession.user_id == user.id).all()
        if not sessions:
            return {"message": "No chat history to delete."}
        for session in sessions:
            db.delete(session)
        db.commit()
        return {"message": "All chat history has been successfully deleted."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Could not delete chat history: {e}")

@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user_for_api)
):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db.delete(session)
    db.commit()
    return