import json
import shutil
from typing import List
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi import Path as FastApiPath
from fastapi.responses import JSONResponse, StreamingResponse

from .. import auth, models
from ..model_manager import model_manager
from ..schemas import InstructionsUpdateRequest, ManualTrainingRequest
from ..settings import settings
from ..training_status import training_status

router = APIRouter(
    prefix="/api",
    tags=["Training"]
)

def _perform_cleanup():
    """Helper function to delete training artifacts."""
    data_dir = Path("data")
    lora_dir = Path("loras/faq-lora")
    
    deleted_items = []
    errors = []

    # 1. Delete cached data
    cache_dir = data_dir / ".cache"
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            deleted_items.append(f"Cache directory: {cache_dir}")
        except Exception as e:
            errors.append(f"Failed to delete cache directory: {e}")

    # 2. Delete uploaded knowledge files and generated dataset
    if data_dir.exists():
        for f in data_dir.iterdir():
            # Keep the instructions file, delete everything else that is a file.
            if f.is_file() and f.name != "bot_instructions.txt":
                try:
                    f.unlink()
                    deleted_items.append(f"Data file: {f.name}")
                except Exception as e:
                    errors.append(f"Failed to delete file {f.name}: {e}")

    # 3. Delete old LoRA adapter
    if lora_dir.exists():
        try:
            shutil.rmtree(lora_dir)
            deleted_items.append(f"LoRA adapter: {lora_dir}")
        except Exception as e:
            errors.append(f"Failed to delete LoRA directory: {e}")
    
    # Always reset the lora_path setting to ensure the base model is loaded on the next request.
    settings.lora_path = None

    return deleted_items, errors

@router.post("/instructions", status_code=200)
async def save_instructions(request_data: InstructionsUpdateRequest, user: models.User = Depends(auth.get_current_user_for_api)):
    """
    Saves the bot's behavioral instructions to a dedicated file.
    """
    instructions = request_data.instructions
    
    instructions_path = Path("data/bot_instructions.txt")
    try:
        instructions_path.parent.mkdir(exist_ok=True)
        instructions_path.write_text(instructions, encoding="utf-8")
        return {"message": "Instructions saved successfully."}
    except Exception as e:
        print(f"Error saving instructions: {e}")
        raise HTTPException(status_code=500, detail="Failed to save instructions.")

@router.get("/instructions", response_class=JSONResponse)
async def get_instructions(user: models.User = Depends(auth.get_current_user_for_api)):
    """
    Reads and returns the current bot instructions.
    """
    instructions_path = Path("data/bot_instructions.txt")
    if instructions_path.exists():
        return {"instructions": instructions_path.read_text(encoding="utf-8")}
    return {"instructions": ""}

@router.get("/train/status")
async def get_training_status(request: Request, user: models.User = Depends(auth.get_current_user_for_api)):
    """
    Server-Sent Events endpoint to stream training status updates.
    """
    # The user dependency ensures this endpoint is protected.
    async def event_stream():
        while True:
            # Check if the client has disconnected to prevent the loop from running forever
            if await request.is_disconnected():
                print("SSE client for training status has disconnected.")
                break

            # Wait for an update from the training process
            await training_status.wait()
            # Create a JSON payload with both message and running status
            status_payload = {
                "message": training_status.message,
                "running": training_status.running
            }
            # Send the update to the client as a JSON string
            yield f"data: {json.dumps(status_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/reload_model", status_code=200)
async def reload_model(_: Request, __: models.User = Depends(auth.get_current_user_for_api)):
    """
    Clears the model cache and reloads the model from disk.
    This is useful for loading a newly fine-tuned LoRA adapter.
    """
    print("Received request to reload model...")
    try:
        model_manager.get_chat_model(force_reload=True)
        return {"message": "Model reloaded successfully."}
    except Exception as e:
        print(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
async def upload_and_train(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    _: models.User = Depends(auth.get_current_user_for_api)
):
    """
    Receives knowledge base files and starts the fine-tuning process.
    """
    from ..training import run_training

    upload_dir = Path("data")
    upload_dir.mkdir(exist_ok=True)
    
    saved_files = []
    for file in files:
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        saved_files.append(file.filename)

    background_tasks.add_task(run_training, skip_behavioral_generation=False)

    return JSONResponse(content={"message": f"Files '{', '.join(saved_files)}' saved. Starting cumulative training."})

@router.post("/train/manual", status_code=200)
async def train_from_manual_dataset(
    request: ManualTrainingRequest,
    background_tasks: BackgroundTasks,
    _: models.User = Depends(auth.get_current_user_for_api)
):
    """
    Receives a manually created dataset, saves it, and starts fine-tuning.
    """
    from ..training import run_training

    dataset_content = request.dataset_content.strip()
    if not dataset_content:
        raise HTTPException(status_code=400, detail="Dataset content cannot be empty.")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    manual_data_path = data_dir / "manual_dataset.jsonl"
    
    try:
        for i, line in enumerate(dataset_content.splitlines()):
            data = json.loads(line)
            if not isinstance(data, dict) or "text" not in data or not isinstance(data["text"], str) or not data["text"].strip():
                raise ValueError(f"Line {i+1} is invalid.")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSONL format: {e}") from e

    manual_data_path.write_text(dataset_content, encoding="utf-8")
    background_tasks.add_task(run_training, skip_behavioral_generation=True)

    return {"message": "Manual dataset added. Starting cumulative training."}

@router.get("/train/files", response_model=List[str])
async def list_training_files(_: models.User = Depends(auth.get_current_user_for_api)):
    """Lists all knowledge files currently in the data directory."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    
    supported_extensions = ['.pdf', '.docx', '.txt', '.jsonl']
    files = [f.name for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions and f.name != "bot_instructions.txt"]
    return sorted(files)

@router.delete("/train/files/{filename}", status_code=200)
async def delete_training_file(
    filename: str = FastApiPath(..., description="The name of the file to delete"),
    _: models.User = Depends(auth.get_current_user_for_api)
):
    """Deletes a specific knowledge file and its corresponding cache entry."""
    data_dir = Path("data").resolve()
    file_path = (data_dir / filename).resolve()

    if not str(file_path).startswith(str(data_dir)) or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename or path.")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    try:
        file_path.unlink()
        cache_path = data_dir / ".cache" / f"{filename}.jsonl"
        if cache_path.exists():
            cache_path.unlink()
        return {"message": f"File '{filename}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not delete file: {e}")

@router.post("/train/clear", status_code=200)
async def clear_training_data(_: models.User = Depends(auth.get_current_user_for_api)):
    """
    Deletes all training data, caches, and the fine-tuned adapter.
    Crucially, it also resets the in-memory model state to ensure a clean start.
    """
    print("--- Received request to clear all training data ---")
    deleted_items, errors = _perform_cleanup()

    if errors:
        raise HTTPException(status_code=500, detail=f"Failed to delete some items: {'; '.join(errors)}")

    print("--- Resetting in-memory model state ---")
    model_manager.reset()

    return {"message": "All previous training data has been cleared.", "deleted": deleted_items}