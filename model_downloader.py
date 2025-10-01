#!/usr/bin/env python3
"""
Model Downloader for Mistral 7B GGUF format and Hugging Face models
"""

import os
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path
import argparse

def download_gguf_model(model_name: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", 
                       file_name: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                       local_dir: str = "models"):
    """
    Download a GGUF model from Hugging Face Hub
    """
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading {model_name}/{file_name}...")
    
    try:
        model_path = hf_hub_download(
            repo_id=model_name,
            filename=file_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"GGUF model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading GGUF model: {e}")
        return None

def download_hf_model(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                     local_dir: str = "models/hf"):
    """
    Download a Hugging Face model for fine-tuning
    """
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading {model_name} for fine-tuning...")
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.gguf", "*.bin", "*.safetensors"]  # We don't need these for GGUF
        )
        print(f"Hugging Face model downloaded to: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"Error downloading Hugging Face model: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models for chatbot")
    parser.add_argument("--gguf-model", type=str, default="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                       help="Hugging Face GGUF model repository ID")
    parser.add_argument("--gguf-file", type=str, default="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                       help="Specific GGUF model file to download")
    parser.add_argument("--hf-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                       help="Hugging Face model for fine-tuning")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save the models")
    
    args = parser.parse_args()
    
    # Download GGUF model for inference
    gguf_path = download_gguf_model(args.gguf_model, args.gguf_file, args.output_dir)
    
    # Download HF model for fine-tuning
    hf_path = download_hf_model(args.hf_model, f"{args.output_dir}/hf")
    
    if gguf_path:
        print("GGUF download completed successfully!")
        print(f"You can now run the chatbot with: python chatbot.py --model-path {gguf_path}")
    else:
        print("GGUF download failed!")
    
    if hf_path:
        print("Hugging Face model download completed successfully!")
        print(f"You can use this for fine-tuning with: python fine_tuner.py")
    else:
        print("Hugging Face model download failed!")