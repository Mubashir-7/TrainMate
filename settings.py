from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Union

class _Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Security & Database ---
    database_url: str = "sqlite:///./app.db"
    secret_key: str = "a_very_secret_key_that_should_be_changed" # Change this in production
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 # 1 day

    # --- Core model info (can be set via .env or environment variables)
    # Default to a lightweight GGUF model, ideal for CPU-only environments.
    # This model is small, fast, and aligns with the default training model.
    # To use the full, un-quantized model for chat, change model_type to "hf"
    # and update the path to the local Hugging Face model directory.
    model_path: str = "models/mistral-7b-instruct-v0.3.Q8_0.gguf"
    model_type: str = "gguf" # 'hf' or 'gguf'
    # Set lora_path to None to run the base model without any fine-tuning.
    lora_path: Union[str, None] = None

    # --- Training ---
    # The base model for fine-tuning. It's the Hugging Face version of our chat model.
    training_base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    # By default, require a GPU for training. Set to false in .env to train on CPU (very slow).
    require_gpu_for_training: bool = False
    # 4-bit quantization is a GPU-specific feature. It should be False for CPU-only training.
    training_use_4bit_quantization: bool = False
    knowledge_examples_per_chunk: int = 1   # Number of Q&A pairs to generate per knowledge chunk
    examples_per_section: int = 2           # Number of Q&A pairs to generate per section in the instruction file
    hf_token: Union[str, None] = None             # HuggingFace Hub token for gated models
    training_synth_temperature: float = 0.8 # Temperature for generating synthetic Q&A pairs
    training_synth_max_tokens: int = 2000    # Max tokens for synthetic data generation responses

    # --- LoRA Config ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # --- Trainer Config (optimized for small datasets of 50-200 examples) ---
    training_epochs: int = 20           # More epochs are needed for small datasets
    training_batch_size: int = 4        # Increase for high-RAM machines
    training_grad_acc_steps: int = 1    # Decrease as batch size increases
    training_lr: float = 2e-6           # A very small learning rate for very small datasets
    training_warmup_ratio: float = 0.05 # Percentage of training steps for warmup
    training_weight_decay: float = 0.01 # Regularization
    training_logging_steps: int = 10    # How often to log training metrics
    training_save_total_limit: int = 2  # The total number of checkpoints to save
    training_dataloader_workers: int = 2 # Number of workers for the dataloader
    training_early_stopping_patience: int = 2 # Stop if eval_loss doesn't improve for this many evaluations.
    training_max_seq_length: int = 2048 # Aligned with model's context window to prevent truncation

    # --- Generation
    active_model: str = "llama" #Can be "llama" or "t5"
    max_tokens: int = 512
    temperature: float = 0.7
    context_size: int = 32768           # GGUF context size, aligned with the training sequence length
    max_history: int = 4                # last n Q‑A pairs kept

    # --- llama-cpp GPU layers (for GGUF models) ---
    # Offloads layers to the GPU. Set to 0 for CPU-only inference.
    # To use a GPU if available, set this to -1 in your .env file.
    gguf_gpu_layers: int = 0

    # --- misc
    slow_tokenizer: bool = False

settings = _Settings()  # singleton

# --- Post-initialization logic ---
# Check if a LoRA adapter exists at the default path and update the setting if so.
# This ensures that a fine-tuned model is loaded automatically on startup if it's available.
default_lora_path = "loras/faq-lora"
# Check for the specific config file to ensure the adapter is valid and complete.
if (Path(default_lora_path) / "adapter_config.json").exists():
    print(f"✅ Found valid LoRA adapter at '{default_lora_path}'. Setting lora_path.")
    settings.lora_path = default_lora_path
else:
    print(f"ℹ️ No valid LoRA adapter found at '{default_lora_path}'. The base model will be used.")
