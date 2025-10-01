#!/bin/bash
# Convenience script to run the enhanced humanized chatbot

# Default model path
MODEL_PATH="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Check if model exists, download if not
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found. Downloading..."
    python model_downloader.py
fi

# Run the enhanced chatbot
python chatbot.py --model-path "$MODEL_PATH" "$@"