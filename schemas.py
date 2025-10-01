from pydantic import BaseModel
from typing import List, Literal, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    history: List[ChatMessage]
    session_id: Optional[int] = None

# --- Chat History Schemas ---

class MessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime
    class Config:
        from_attributes = True

class SessionOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    class Config:
        from_attributes = True

class UserProfileUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    xelence_x_api_key: Optional[str] = None
    xelence_affiliateid: Optional[str] = None
    chat_rate: Optional[float] = None
    new_password: Optional[str] = None
    confirm_password: Optional[str] = None

class ManualTrainingRequest(BaseModel):
    dataset_content: str

class InstructionsUpdateRequest(BaseModel):
    instructions: str