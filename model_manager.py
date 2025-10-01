import gc
import torch
from pathlib import Path

from .training_status import training_status
from .settings import settings
from . import model_loader # Use the existing loader functions

# These imports are needed for the new logic
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

class ModelManager:
    """
    A singleton class to manage a single, persistent model in memory.
    The model is loaded once at startup and then reconfigured for either
    chat (inference) or training.
    """
    def __init__(self):
        self._base_model = None      # For HF training model
        self._chat_model = None      # For the active chat model (GGUF or HF+LoRA)
        self._tokenizer = None
        print("🤖 ModelManager initialized.")

    def _ensure_clean_base_model(self):
        """
        A private helper to robustly clean the base model of any PEFT modifications.
        This is called before any operation that requires the pristine base model,
        ensuring a clean state and preventing PEFT warnings.
        """
        if self._base_model is None:
            return

        if isinstance(self._base_model, PeftModel):
            print("💡 Unloading PEFT adapter to get the clean base model.")
            self._base_model = self._base_model.unload()
            # It's good practice to hint garbage collection after big object changes
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if hasattr(self._base_model, "peft_config"):
            print("🗑️ Removing lingering 'peft_config' attribute from base model.")
            delattr(self._base_model, "peft_config")
        if hasattr(self._base_model, "is_peft_prepared"):
            print("🗑️ Removing lingering 'is_peft_prepared' attribute from base model.")
            delattr(self._base_model, "is_peft_prepared")

    def load_model_at_startup(self):
        """
        Loads the base Hugging Face model and tokenizer into memory. This should only be
        called when an HF model is needed (for training or HF chat).
        """
        if self._base_model is not None:
            print("✅ Base HF model is already loaded. Skipping.")
            return

        print("--- Loading base HF model into memory for persistent use ---")
        training_status.set("Loading base HF model...", True)
        
        from .model_loading_utils import load_huggingface_model_for_training
        
        self._base_model, self._tokenizer = load_huggingface_model_for_training()
        
        print("✅ Base HF model loaded successfully.")
        training_status.set("Idle", False)

    def get_chat_model(self, force_reload: bool = False):
        """
        Returns the model configured for chat based on settings.
        This can be a GGUF model or a Hugging Face model (with/without LoRA).
        """
        print("➡️ Request received for CHAT model.")

        if not force_reload and self._chat_model:
            print("✅ Returning cached chat model.")
            return self._chat_model, self._tokenizer

        if force_reload:
            print("🔄 Forcing reload of chat model.")
            self._chat_model = None
            # The model_loader cache will also be cleared by its own logic.

        training_status.set("Loading chat model...", True)

        # The model_loader is now the single source of truth for loading chat models.
        # It intelligently determines whether to load a GGUF or HF model based on the
        # `model_path` setting, making the system more robust.
        print("💡 Delegating chat model loading to model_loader...")
        self._chat_model, self._tokenizer = model_loader.get_model(force_reload=force_reload)
        print("✅ Chat model received from loader.")

        training_status.set("Idle", False)
        return self._chat_model, self._tokenizer

    def get_training_model(self):
        """
        Returns the persistent base model, ensuring it's unwrapped and ready for training.
        """
        print("➡️ Request received for TRAINING model.")
        # If model is not loaded (e.g., after a reset), load it now.
        if self._base_model is None:
            print("🚨 Base model not found in memory. Triggering load...")
            self.load_model_at_startup()
        
        self._ensure_clean_base_model()

        # Ensure the model is in training mode and cache is disabled
        self._base_model.train()
        self._base_model.config.use_cache = False
        
        print("✅ Returning persistent base model configured for training.")
        return self._base_model, self._tokenizer

    def reset(self):
        """
        Resets the entire state of the manager, unloading models from memory.
        This is the most effective way to ensure a clean state and prevent
        PEFT warnings on subsequent operations.
        """
        print("🚨 Resetting ModelManager state...")
        # By setting all model references to None, we allow Python's garbage
        # collector to reclaim the memory. There's no need to manually unload
        # adapters if we are discarding the objects entirely.
        self._base_model = None
        self._chat_model = None
        self._tokenizer = None
        # Force garbage collection to release VRAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ ModelManager state has been reset. The model will be reloaded on the next request.")

model_manager = ModelManager()