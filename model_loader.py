"""
Lightweight model/LoRA loader with on‑demand singleton cache.
Supported base types:
  • 'hf'        → transformers AutoModel
  • 'gguf'      → llama‑cpp‑python (CPU/GPU)
"""
from functools import lru_cache
from typing import Iterator, List, Dict
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
import torch
from huggingface_hub import snapshot_download

try:
    from peft import PeftModel, prepare_model_for_kbit_training
except ImportError:
    PeftModel = None  # optional
    prepare_model_for_kbit_training = None

from .settings import settings

def _load_t5_model():
    """Loads the T5 model and tokenizer."""
    model_name = "google/flan-t5-small"
    print(f"--- Loading T5 model: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def _load_gguf_model():
    """Loads a GGUF model using llama-cpp-python."""
    from llama_cpp import Llama
    print(f"--- Loading GGUF model: {settings.model_path} ---")
    llm = Llama(
        model_path=settings.model_path,
        n_gpu_layers=settings.gguf_gpu_layers,
        n_ctx=settings.context_size, # type: ignore
        chat_format="mistral-instruct", # Mistral Instruct models use this format
        add_bos=True,
    )
    tokenizer = None  # llama‑cpp handles its own tokenisation
    return llm, tokenizer

def _load_hf_model():
    """Loads a full Hugging Face model and applies a LoRA adapter if specified."""
    model_id_or_path = settings.model_path
    safe_model_name = model_id_or_path.replace("/", "__")
    model_source_path = Path("models/hf") / safe_model_name

    if model_source_path.is_file():
        raise TypeError(
            f"Configuration Error: The model_type is set to 'hf', which requires a directory, "
            f"but the model_path ('{model_source_path}') points to a single file. \n\n"
            f"To fix this, either set MODEL_TYPE='gguf' in your settings or .env file to use this GGUF file, "
            f"or update MODEL_PATH to point to the correct Hugging Face model directory."
        )

    if not model_source_path.exists() or not any(model_source_path.iterdir()):
        print(f"⚠️ Local model cache not found at '{model_source_path}'.")
        print(f"Attempting to download from Hugging Face Hub: {model_id_or_path}")
        model_source_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id_or_path,
            local_dir=str(model_source_path),
            local_dir_use_symlinks=False,
        )
        print(f"✅ Model downloaded to '{model_source_path}'.")
    else:
        print(f"✅ Found local model cache at '{model_source_path}'. Loading from disk.")

    print(f"--- Loading Hugging Face model from: {model_source_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(str(model_source_path), use_fast=not settings.slow_tokenizer)
    if tokenizer.pad_token is None:
        print("⚠️ Tokenizer does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Load the base model in standard half-precision. This provides a stable
    # structure for the LoRA adapter to be applied, avoiding KeyErrors caused
    # by inconsistencies in quantization during inference.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_source_path),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if settings.lora_path and PeftModel:
        lora_path = Path(settings.lora_path)
        if lora_path.exists():
            print(f"🚀 Applying LoRA adapter from '{settings.lora_path}'...")
            model = PeftModel.from_pretrained(model, str(lora_path))
        else:
            print(f"⚠️ Warning: LoRA path '{settings.lora_path}' not found. Using base model without adapter.")

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer

@lru_cache(maxsize=1)
def _get_model_cached():
    """The actual cached model loading function."""
    print("--- Loading model from disk (cache miss) ---")

    if settings.active_model == "t5": # This is a legacy option
        return _load_t5_model()

    # The primary logic is now based on the explicit `model_type` setting for clarity.
    if settings.model_type == "gguf":
        print("💡 Model type is 'gguf'. Loading GGUF model.")
        return _load_gguf_model()
    elif settings.model_type == "hf":
        print("💡 Model type is 'hf'. Loading Hugging Face model.")
        return _load_hf_model()
    else:
        raise ValueError(
            f"Configuration Error: Invalid 'model_type' in settings: '{settings.model_type}'. Please use 'gguf' or 'hf'."
        )

def get_model(force_reload: bool = False):
    """
    Wrapper around the cached model loader to allow for forced reloads.
    """
    if force_reload:
        _get_model_cached.cache_clear()
    return _get_model_cached()


def generate_stream(
    model,
    tokenizer,
    history: List[Dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    system_prompt: str = None,
) -> Iterator[str]:
    """
    Yields token by token response from the model.
    Works for both GGUF (llama-cpp) and Hugging Face transformers.
    """
    if tokenizer is None:  # gguf / llama‑cpp
        # The user's sample script shows that manually building the prompt string
        # is the reliable way to apply the system instructions. We will replicate that here.
        
        # The last message in the history is the current user's prompt.
        user_prompt = history[-1]['content'] if history else ''

        # If a system prompt is provided, build the full prompt in the Mistral format.
        # Otherwise, just use the user's prompt.
        if system_prompt:
            full_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        else:
            # Fallback for when there are no instructions.
            full_prompt = f"[INST] {user_prompt} [/INST]"

        print("--- Manually Constructed GGUF Prompt ---")
        print(full_prompt)
        print("-----------------------------------------")

        # Use the model's direct call method with streaming.
        stream = model(
            full_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=["</s>", "[/INST]"], # Stop generation at the end of the model's turn.
            repeat_penalty=1.1,
            stream=True,
        )
        for output in stream:
            if (text_chunk := output["choices"][0].get("text")) is not None:
                yield text_chunk
        return

    # --- Fallback and HF Model Logic ---

    # Prepare the message history for the model.
    # For HF models, we may need to inject a system prompt.
    messages = history
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
        print(f"✅ System prompt injected for HF model.")

    # --- Generation logic for the tiny T5 model ---
    if settings.active_model == "t5":
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        yield generated_text
        return

    # --- Streaming logic for Hugging Face Llama models ---
    lora_status = "fine-tuned" if settings.lora_path and Path(settings.lora_path).exists() else "base"
    print(f"🧠 Using tokenizer's chat template for {lora_status} model.")

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\n" + "="*20 + " FINAL PROMPT TO MODEL " + "="*20)
    print(prompt)
    print("="*51 + "\n")

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.inference_mode()(model.generate)(**gen_kwargs)  # kick off in bg thread

    for new_text in streamer:
        if new_text:  # a streamer can yield empty strings
            yield new_text

