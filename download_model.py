import os
from huggingface_hub import hf_hub_download, snapshot_download, login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# GGUF version of the primary model for fast CPU chatting
GGUF_REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.3-GGUF"
GGUF_FILENAME = "mistral-7b-instruct-v0.3.Q8_0.gguf"
GGUF_LOCAL_DIR = "models"

# The primary Hugging Face model for both chat and fine-tuning
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# This local path must match the convention used in `app/training.py`
HF_LOCAL_DIR = f"models/hf/{HF_MODEL_ID.replace('/', '__')}"

# A token is not required for the public TinyLlama model.
# ---

def download_gguf():
    """Downloads the GGUF model used for chatting."""
    print(f"--- 1/2: Downloading GGUF Model for Chatting ---")
    print(f"Repo: {GGUF_REPO_ID}, File: {GGUF_FILENAME}")
    os.makedirs(GGUF_LOCAL_DIR, exist_ok=True)
    hf_hub_download(
        repo_id=GGUF_REPO_ID,
        filename=GGUF_FILENAME,
        local_dir=GGUF_LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"✅ GGUF model downloaded to '{os.path.join(GGUF_LOCAL_DIR, GGUF_FILENAME)}'\n")

def download_hf_model():
    """Downloads the full Hugging Face model required for fine-tuning."""
    print(f"--- 2/2: Downloading Hugging Face Model for Fine-Tuning ---")
    print(f"Repo: {HF_MODEL_ID}")

    # Check for Hugging Face token for gated models like Mistral or Llama
    hf_token = os.getenv("HF_TOKEN")
    if "mistralai/" in HF_MODEL_ID or "meta-llama/" in HF_MODEL_ID:
        if hf_token:
            print("🔑 Hugging Face token found. Logging in...")
            login(token=hf_token, add_to_git_credential=False)
        else:
            print("\n" + "─" * 70)
            print("⚠️  WARNING: Gated Model Detected")
            print(f"You are trying to download '{HF_MODEL_ID}', which may be a gated model.")
            print("Please create a .env file with your Hugging Face token:")
            print("HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print("You must also accept the license on the model's page on Hugging Face Hub.")
            print("The script will attempt to download, but may fail if you are not logged in.")
            print("─" * 70 + "\n")

    print(f"Starting download of '{HF_MODEL_ID}' to '{HF_LOCAL_DIR}'... (This may take several minutes)")
    os.makedirs(HF_LOCAL_DIR, exist_ok=True)
    try:
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=HF_LOCAL_DIR,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        if "401" in str(e) or "Repository Not Found" in str(e):
            print("\n" + "─" * 70)
            print("❌ Critical Download Error: Repository access denied (401).")
            print("This confirms you need to take action for this gated model.")
            print(f"1. Go to the model page: https://huggingface.co/{HF_MODEL_ID}")
            print("2. Log in and accept the license terms.")
            print("3. Ensure your HF_TOKEN in the .env file is correct and has 'read' permissions.")
            print("4. Run this script again.")
            print("─" * 70 + "\n")
        raise e
    print(f"✅ Hugging Face model downloaded to '{HF_LOCAL_DIR}'\n")

if __name__ == "__main__":
    download_gguf()
    download_hf_model()
    print("🎉 All models have been downloaded successfully!")