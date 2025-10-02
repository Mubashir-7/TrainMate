🧠 TrainMate – One-Stop Edge-Hosted AI Chatbot Platform

TrainMate is a fully customizable, edge-hosted chatbot solution built with FastAPI. It learns directly from your data (PDFs, DOCX, JSONL, and more) to deliver precise, context-aware answers to user queries. With optional fine-tuning, scalable APIs, and database integration, TrainMate is designed to power customer support bots, internal knowledge assistants, and data-driven conversational tools — all running securely on your own machine.

🚀 Features

🧠 Data-Driven Intelligence: TrainMate learns from your documents and knowledge base to provide relevant, high-accuracy answers.

⚙️ Customizable & Extensible: Easily adapt it for FAQs, support, internal knowledge bots, or domain-specific use cases.

🔐 Edge-Hosted Deployment: Runs locally on your machine or server — keeping data under your control.

🧩 Optional Fine-Tuning: Improve domain-specific performance with lightweight fine-tuning workflows.

🗂️ Document Parsing: Supports PDFs, DOCX, JSONL, and more.

🛠️ Production-Ready: Includes JWT-based auth, database ORM, and a user management CLI.

📊 Admin Dashboard: Simple browser-based management panel at /dashboard.

🧰 Tech Stack

Backend: FastAPI + Uvicorn

AI & NLP: Transformers, Torch, PEFT, Accelerate

Data Processing: Datasets, NumPy, SentencePiece

Quantization & Optimization: Bitsandbytes, llama-cpp-python

Document Parsing: PyPDF, python-docx, xmltodict

Database: SQLAlchemy ORM

Security: JWT (python-jose), Passlib

Frontend Templating: Jinja2

Auth & Sessions: Itsdangerous

Deployment: Docker + NVIDIA GPU support (optional)

📦 Requirements

Python 3.10+

(Optional) NVIDIA GPU + CUDA Toolkit for accelerated inference

Docker & Docker Compose (recommended)

🧪 Quick Start
1. Clone the Repository
git clone https://github.com/your-org/trainmate.git
cd trainmate

2. Download Models

Before running, download the default chatbot model:

python download_model.py


📁 This will fetch the base model for inference and fine-tuning.

3. Run with Docker (Recommended)

The easiest way to get started:

docker-compose up --build


Then visit: http://localhost:8000

To run detached:

docker-compose up --build -d

4. Run Locally (Development Mode)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --reload-exclude ".venv/*"


Server available at: http://localhost:8000

🐳 Manual Docker Setup (Advanced)
Step 1: Launch Container

With GPU:

docker run --rm -it --gpus all -v "$(pwd)":/app -p 8000:8000 nvidia/cuda:12.3.2-devel-ubuntu22.04 /bin/bash


With CPU:

docker run -it -v "$(pwd)":/app -p 8000:8000 --name trainmate-container python:3.10-slim /bin/bash

Step 2: Install System Dependencies

Inside the container:

apt-get update && apt-get install -y python3-venv build-essential lsof

Step 3: Create Virtual Environment
cd /app
python3 -m venv .venv
source .venv/bin/activate

Step 4: Install Python Dependencies

GPU Build (with cuBLAS):

export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
export FORCE_CMAKE=1
pip install --no-cache-dir -r /app/requirements.txt


CPU Build:

pip install --no-cache-dir -r /app/requirements.txt

Step 5: Run the Server
uvicorn app.main:app --host 0.0.0.0 --port 8000

🧑‍💻 User Management
Create a User
python create_user.py

Reset Password (Emergency)
python update_user.py

📁 Training Data Format

If you plan to fine-tune TrainMate or extend its knowledge base, use JSON Lines (.jsonl) format:

Each line is a single JSON object.

Each object must contain a text key.

✅ Example:

{"text": "How to reset my password?"}
{"text": "Our support hours are 9 AM – 5 PM, Mon–Fri."}

🔍 Testing the Chatbot
python3 test_model.py \
  --system-prompt system_prompt.txt \
  --chat

📊 Access Points

Main Chat Interface: http://localhost:8000

Admin Dashboard: http://localhost:8000/dashboard

🛡️ Security Notes

Always set a strong SECRET_KEY in your environment variables.

Use HTTPS and a reverse proxy (e.g., Nginx) in production.

Rotate JWT tokens periodically.

🧑‍💼 License

This project is licensed under the MIT License — feel free to use, modify, and deploy it for commercial or personal projects.

📣 Contributing

Pull requests are welcome! Please open an issue before submitting major changes.

🌟 Final Notes

TrainMate is built to give you full control over your AI chatbot — right at the edge. It runs directly on your local machine or private server, learning from your data and delivering accurate, domain-specific answers in real-time.
