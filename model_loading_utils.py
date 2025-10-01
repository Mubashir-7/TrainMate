from pathlib import Path

import torch
from huggingface_hub import login, snapshot_download
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .settings import settings


def load_huggingface_model_for_training():
    """
    Loads the base HuggingFace model and tokenizer.
    It will download the model from Hugging Face Hub on the first run and
    cache it locally for all subsequent runs.
    """
    if not torch.cuda.is_available():
        if settings.require_gpu_for_training:
            raise RuntimeError(
                "Training requires a CUDA-enabled GPU. To override, set REQUIRE_GPU_FOR_TRAINING=false in your .env file."
            )
        print(
            "⚠️ WARNING :: No CUDA-enabled GPU found. Training will proceed on the CPU and will be extremely slow."
        )

    model_id = settings.training_base_model
    safe_model_name = model_id.replace("/", "__")
    local_model_path = Path("models/hf") / safe_model_name

    if not local_model_path.exists() or not any(local_model_path.iterdir()):
        print(f"⚠️ Local training model cache not found at '{local_model_path}'.")
        print(f"Attempting to download from Hugging Face Hub: {model_id}")

        if settings.hf_token:
            print("🔑 Hugging Face token found. Attempting to log in...")
            try:
                login(token=settings.hf_token)
                print("✅ Successfully logged in to Hugging Face Hub.")
            except Exception as e:
                print(f"❌ Failed to log in to Hugging Face Hub: {e}")
                raise RuntimeError(
                    "Hugging Face login failed. Please check if your HF_TOKEN is valid."
                ) from e
        else:
            if "meta-llama" in model_id or "zephyr" in model_id:
                print(
                    f"⚠️ Warning: You are trying to download '{model_id}' which may be a gated model, but no HF_TOKEN was found. Download may fail if you have not accepted the license agreement."
                )

        local_model_path.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_model_path),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            if "Repository Not Found" in str(e) or "401" in str(e) or "404" in str(e):
                print("\n" + "─" * 70)
                print("❌ Critical Download Error: The repository could not be accessed.")
                print(
                    "This often happens with gated models like Llama or Zephyr that require you to accept a license agreement on the Hugging Face website."
                )
                print("\n**Action Required:**")
                print(f"1. Go to the model page: https://huggingface.co/{model_id}")
                print("2. Log in with the account associated with your HF_TOKEN.")
                print("3. Click the button to accept the license terms and access the repository.")
                print("4. Ensure your HF_TOKEN in the .env file is correct and has 'read' permissions.")
                print("5. Run the training again.")
                print("─" * 70 + "\n")
            raise e
        print(f"✅ Model downloaded to '{local_model_path}'.")
    else:
        print(f"✅ Found local training model cache at '{local_model_path}'. Loading from disk.")

    print(f"--- Loading base model for training from: {local_model_path} ---")

    tokenizer = AutoTokenizer.from_pretrained(
        str(local_model_path), use_fast=not settings.slow_tokenizer
    )
    if tokenizer.pad_token is None:
        print("⚠️ Tokenizer does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    model_load_kwargs = {"device_map": "auto"}
    if settings.training_use_4bit_quantization:
        print("💪 Using 4-bit quantization for memory efficiency.")
        model_load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_load_kwargs["low_cpu_mem_usage"] = True
    else:
        model_load_kwargs["torch_dtype"] = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

    model = AutoModelForCausalLM.from_pretrained(str(local_model_path), **model_load_kwargs)

    model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if settings.training_use_4bit_quantization:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer