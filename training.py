import time
import os
import re

# Set this environment variable before any tokenizers are used.
# This prevents a harmless but noisy warning when using multiple dataloader workers.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import json
import gc
from typing import Union, List, Dict
import importlib.metadata

# For reading uploaded files
from pypdf import PdfReader
from docx import Document

# For actual training
import torch
from datasets import Dataset
from transformers import TrainingArguments, EarlyStoppingCallback, trainer_utils
from peft import LoraConfig
from trl import SFTTrainer


from .settings import settings
from .training_status import training_status

# --- 1. File Parsing ---

def read_uploaded_file(file_path: str) -> str:
    """
    Reads the content of an uploaded .txt, .pdf, or .docx file.
    """
    path = Path(file_path)
    if path.suffix == ".pdf":
        reader = PdfReader(path)
        return "\n\n".join([page.extract_text() for page in reader.pages])
    elif path.suffix == ".docx":
        doc = Document(path)
        return "\n\n".join([para.text for para in doc.paragraphs])
    elif path.suffix == ".txt":
        return path.read_text()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

# --- 2. Instruction Parsing ---

def parse_instructions(raw_text: str) -> dict:
    """
    Parses the raw text from the instruction file into a dictionary,
    using '#' as the section delimiter.
    """
    print("🧠 Parsing instruction file into structured data...")
    instructions = {}
    current_key = None
    content_lines = []

    def save_section(key, content):
        if key and content:
            instructions[key] = "\n".join(content).strip()

    for line in raw_text.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith('#'):
            # Save the previous section before starting a new one
            save_section(current_key, content_lines)
            
            # Start a new section by cleaning the heading
            current_key = stripped_line.lstrip('#').strip()
            content_lines = []
        elif current_key:
            # Add content to the current section
            content_lines.append(line)

    # Save the very last section after the loop finishes
    save_section(current_key, content_lines)

    if instructions:
        print(f"✅ Parsed {len(instructions)} sections: {list(instructions.keys())}")
    else:
        print("⚠️ Warning: No sections found. The instruction file might be empty or missing '#' headings.")
    return instructions

# --- 3. Synthetic Data Generation ---

def _is_likely_code(text: str) -> bool:
    """A simple heuristic to check if a string is likely code."""
    # Keywords, symbols, and patterns common in code but not natural language
    code_indicators = [
        'const ', 'let ', 'var ', 'import ', 'from ', '=>', 'useEffect',
        'useState', '<div>', '</div>', 'class=', 'className=', 'document.getElementById',
        'async function', 'def ', 'public static', 'private void',
        '//', '/*', '*/',
    ]
    # Check for presence of multiple indicators
    indicator_count = sum(1 for indicator in code_indicators if indicator in text)

    # Check if the text is a valid JSON object, which is a strong indicator of code.
    # This is more robust than just checking for braces.
    is_json_object = False
    stripped_text = text.strip()
    if stripped_text.startswith('{') and stripped_text.endswith('}'):
        try:
            # Attempt to parse the text as JSON. If it succeeds, it's likely code.
            json.loads(stripped_text)
            is_json_object = True
        except json.JSONDecodeError:
            # If it fails to parse, it's probably not a JSON object.
            is_json_object = False
    
    # If it has multiple code words or is a valid JSON object, flag it.
    return indicator_count >= 2 or is_json_object

def _clean_generated_text(text: str) -> str:
    """
    Cleans the raw output from the model during synthetic data generation.
    Removes common artifacts like prompt remnants and conversational filler.
    """
    # Remove markdown-style artifacts like code blocks, bolding, and italics
    text = re.sub(r'```[\w\s]*\n?', '', text) # Handle optional newline
    text = re.sub(r'```', '', text)
    text = text.replace('**', '')
    text = text.replace('*', '') # Remove italics markers

    # Remove parenthetical answers like (Answer: ...) that can appear in questions
    text = re.sub(r'\s*\([^)]*Answer:[^)]*\)', '', text, flags=re.IGNORECASE)

    # Remove markdown horizontal rules that can appear before the response
    text = re.sub(r'^\s*---\s*', '', text.strip(), flags=re.MULTILINE).strip()

    # Define a list of filler phrases to remove from the beginning of the text
    fillers = [
        "sure, here's the user's question", "sure, here's the question", "here is the user's question",
        "here's the user question", "user question", "question",
        "sure, here's the response", "sure, here's the answer", "here is the response",
        "here's the response", "response", "answer", "bot", "user",
        # New fillers found from user's logs
        "bot's response", "perfect response", "your factual answer", "factual answer",
        "task", "one question that arises from the given text is",
        # Persona name as a prefix
        "saad el soussi",
        # New artifact from latest logs
        "'s response"
    ]
    
    # Create a regex pattern to match any of the fillers at the start of the string,
    # case-insensitively, followed by optional colons and whitespace.
    # This is more robust than looping with startswith.
    # The `\s*` handles spaces, `[:\s]*` handles colons and more spaces.
    fillers_pattern = '|'.join(re.escape(f) for f in fillers)
    pattern = re.compile(r'^\s*(?:' + fillers_pattern + r')\s*[:\s]*', re.IGNORECASE)
    
    # Substitute the found pattern with an empty string
    cleaned_text = pattern.sub('', text.strip())

    # Final trim of whitespace and quotes
    final_text = cleaned_text.strip().strip('"`')

    # Also remove emojis from the generated text to prevent the model from learning to use them.
    # This regex covers most of the emoji characters.
    return re.sub(r'[\U00010000-\U0010ffff]', '', final_text)

def _calculate_max_generation_tokens(model, tokenizer, question: str, response_prompt_len: int) -> int:
    """
    Calculates the maximum number of new tokens that can be generated for a response,
    considering multiple constraints to prevent errors.
    """
    # 1. Calculate space available in the final training sequence
    question_tokens = tokenizer(question, return_tensors="pt")["input_ids"]
    # Overhead for special tokens in the template and a safety margin
    template_overhead = 20
    max_len_for_training = settings.training_max_seq_length - question_tokens.shape[1] - template_overhead

    # 2. Calculate space available in the generation model's context window
    # Use the model's actual context window size for accuracy
    model_context_window = getattr(model.config, 'max_position_embeddings', settings.training_max_seq_length)
    available_space_for_gen = model_context_window - response_prompt_len - template_overhead

    # 3. The final limit is the minimum of all constraints
    max_tokens = min(
        settings.training_synth_max_tokens, # User-defined absolute max
        max_len_for_training,             # To prevent truncation in trainer
        available_space_for_gen           # To prevent exceeding model context
    )
    return max(0, max_tokens) # Ensure we don't return a negative number

def generate_behavioral_dataset(instructions: dict, model, tokenizer) -> List[Dict]:
    """
    Generates a dataset based on behavioral instructions (system role).
    It creates a few examples for each instruction section to teach the model
    *how to behave*.
    """
    print("🤖 Generating behavioral dataset from instructions...")
    dataset = []
    num_sections = len(instructions)

    for i, (section_title, section_content) in enumerate(instructions.items()):
        for j in range(settings.examples_per_section):
            status_msg = f"Section {i+1}/{num_sections} ('{section_title}'): Generating example {j+1}/{settings.examples_per_section}..."
            training_status.set(status_msg, True)

            # --- Step 1: Generate a realistic user question with a retry mechanism ---
            question = ""
            language_hint = ""
            if "arabic" in section_content.lower():
                language_hint = "The user question MUST be in Arabic."
            elif "spanish" in section_content.lower():
                language_hint = "The user question MUST be in Spanish."

            question_prompt_history = [
                {"role": "user", "content": f"Here is a rule for a chatbot:\n---\n{section_content}\n---\nWrite a single, common question from a user that would be handled by this rule.\nFor example, if the rule is about handling signups, a good question is \"How do I sign up?\".\nIf the rule is about greetings, a good question is \"hello\".\nYour question should be short and simple. {language_hint}"}
            ]
            question_prompt = tokenizer.apply_chat_template(
                question_prompt_history,
                tokenize=False,
                add_generation_prompt=True
            )

            for attempt in range(3): # Retry up to 3 times
                with torch.no_grad():
                    input_tensors = tokenizer(question_prompt, return_tensors="pt")
                    inputs = {key: value.to(model.device) for key, value in input_tensors.items()}
                    # This input_length is specific to this generation call
                    input_length = inputs["input_ids"].shape[1]
                    outputs = model.generate(**inputs, max_new_tokens=50, temperature=settings.training_synth_temperature, do_sample=True)
                    generated_tokens = outputs[0][input_length:]
                    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    question = _clean_generated_text(raw_text)
                
                if question and len(question.split()) >= 3:
                    break # Successfully generated a valid question
                else:
                    print(f"   - Retry {attempt+1}/3: Generated question was too short. Retrying...")
                    time.sleep(0.5) # Small delay before retrying

            # If after retries the question is still invalid, skip this example
            if not question or len(question.split()) < 3:
                print(f"   - Warning: Failed to generate a valid question for section '{section_title}' after 3 attempts. Skipping.")
                continue

            # --- Step 2: Generate a perfectly-styled response to that question ---
            response_prompt_history = [
                {"role": "system", "content": section_content},
                {"role": "user", "content": question}
            ]
            response_prompt = tokenizer.apply_chat_template(response_prompt_history, tokenize=False, add_generation_prompt=True)

            print("\n" + "-"*20 + f" PERSONA FOR SECTION '{section_title}' " + "-"*20)
            print(section_content)
            print("-" * (43 + len(section_title)))
            print(f"\n--- Generating response for section '{section_title}' ---")
            print(f"   - Using prompt to enforce persona:\n{response_prompt}")

            with torch.no_grad():
                response_prompt_tokens = tokenizer(response_prompt, return_tensors="pt")
                inputs = {key: value.to(model.device) for key, value in response_prompt_tokens.items()}
                prompt_len_for_gen = inputs["input_ids"].shape[1]

                max_generation_tokens = _calculate_max_generation_tokens(
                    model, tokenizer, question, prompt_len_for_gen
                )

                if max_generation_tokens <= 0:
                    print(f"   - Warning: Prompt for section '{section_title}' is too long to generate a response. Skipping.")
                    continue

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_generation_tokens,
                    temperature=settings.training_synth_temperature,
                    do_sample=True
                )
                generated_tokens = outputs[0][prompt_len_for_gen:]
                raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                response = _clean_generated_text(raw_text)

            # Add a filter to prevent training on generated code snippets
            if _is_likely_code(response):
                print(f"   - Warning: Generated response for section '{section_title}' appears to be code. Skipping.")
                continue

            # Also check for trivially short responses
            if response and len(response.split()) >= 3:
                # Create the final training example in ChatML format
                training_example = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]
                # The SFTTrainer's formatting_func expects a 'text' field containing the full formatted string.
                # Do not add a generation prompt here, as the example is complete.
                formatted_text = tokenizer.apply_chat_template(training_example, tokenize=False, add_generation_prompt=False)
                dataset.append({"text": formatted_text})
                print(f"   - ✅ Generated Q&A pair:\n     Q: {question}\n     A: {response}")
            else:
                print(f"   - Warning: Could not generate a response for section '{section_title}'. Skipping.")
                continue

    print(f"✅ Generated {len(dataset)} synthetic Q&A pairs.")
    return dataset

def generate_knowledge_dataset(knowledge_text: str, model, tokenizer) -> List[Dict]:
    """
    Generates a dataset from a raw knowledge base text.
    It splits the text into smaller chunks and generates a Q&A pair for each
    chunk to teach the model *what to know*.
    """
    print("🧠 Generating knowledge dataset from raw text...")
    dataset = []

    # Split text into chunks by paragraphs. Filter out empty lines.
    chunks = [chunk.strip() for chunk in knowledge_text.split('\n\n') if chunk.strip()]
    num_chunks = len(chunks)
    print(f"   - Split knowledge base into {num_chunks} chunks.")

    for i, chunk in enumerate(chunks):
        # For each chunk, generate the configured number of Q&A pairs.
        for j in range(settings.knowledge_examples_per_chunk):
            status_msg = f"Knowledge Chunk {i+1}/{num_chunks}: Generating example {j+1}/{settings.knowledge_examples_per_chunk}..."
            training_status.set(status_msg, True)

            # --- Step 1: Generate a relevant question with a retry mechanism ---
            question = ""
            question_prompt_history = [
                {"role": "user", "content": f"Here is a piece of text:\n---\n{chunk}\n---\nWrite a single, direct question that a user would ask to get a specific piece of information from this text.\nFor example, if the text says \"Our office is open from 9 AM to 5 PM\", a good question is \"What are your office hours?\".\nDo not ask for a summary. Ask a specific question."}
            ]
            question_prompt = tokenizer.apply_chat_template(
                question_prompt_history,
                tokenize=False,
                add_generation_prompt=True
            )

            for attempt in range(3): # Retry up to 3 times
                with torch.no_grad():
                    input_tensors = tokenizer(question_prompt, return_tensors="pt")
                    inputs = {key: value.to(model.device) for key, value in input_tensors.items()}
                    # This input_length is specific to this generation call
                    input_length = inputs["input_ids"].shape[1]
                    outputs = model.generate(**inputs, max_new_tokens=50, temperature=settings.training_synth_temperature, do_sample=True)
                    generated_tokens = outputs[0][input_length:]
                    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    question = _clean_generated_text(raw_text)

                if question and len(question.split()) >= 3:
                    break # Successfully generated a valid question
                else:
                    print(f"   - Retry {attempt+1}/3: Generated question for chunk {i+1} was too short. Retrying...")
                    time.sleep(0.5) # Small delay before retrying

            # If after retries the question is still invalid, skip this chunk
            if not question or len(question.split()) < 3:
                print(f"   - Warning: Failed to generate a valid question for chunk {i+1} after 3 attempts. Skipping.")
                continue

            # --- Step 2: Generate a factual answer ---
            response_prompt_history = [
                {"role": "system", "content": f"Use the following context to answer the question.\n---Context: \"{chunk}\"\n---"},
                {"role": "user", "content": question}
            ]
            response_prompt = tokenizer.apply_chat_template(response_prompt_history, tokenize=False, add_generation_prompt=True)

            print(f"\n--- Generating answer for knowledge chunk {i+1} ---")
            print(f"   - Using prompt to extract knowledge:\n{response_prompt}")

            with torch.no_grad():
                response_prompt_tokens = tokenizer(response_prompt, return_tensors="pt")
                inputs = {key: value.to(model.device) for key, value in response_prompt_tokens.items()}
                prompt_len_for_gen = inputs["input_ids"].shape[1]

                max_generation_tokens = _calculate_max_generation_tokens(
                    model, tokenizer, question, prompt_len_for_gen
                )

                if max_generation_tokens <= 0:
                    print(f"   - Warning: Prompt for chunk {i+1} is too long to generate a response. Skipping.")
                    continue

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_generation_tokens,
                    temperature=settings.training_synth_temperature,
                    do_sample=True
                )
                generated_tokens = outputs[0][prompt_len_for_gen:]
                raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                response = _clean_generated_text(raw_text)

            # Add a filter to prevent training on generated code snippets
            if _is_likely_code(response):
                print(f"   - Warning: Generated response for chunk {i+1} appears to be code. Skipping.")
                continue

            # Also check for trivially short responses
            if response and len(response.split()) >= 3:
                # Create the final training example in ChatML format
                training_example = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]
                # The SFTTrainer's formatting_func expects a 'text' field containing the full formatted string.
                # Do not add a generation prompt here, as the example is complete.
                formatted_text = tokenizer.apply_chat_template(training_example, tokenize=False, add_generation_prompt=False)
                dataset.append({"text": formatted_text})
                print(f"   - ✅ Generated Q&A pair:\n     Q: {question}\n     A: {response}")
            else:
                print(f"   - Warning: Could not generate an answer for chunk {i+1}. Skipping.")
                continue

    print(f"✅ Generated {len(dataset)} Q&A pairs from the knowledge base.")
    return dataset


# --- 4. Real Training Implementation ---

def perform_fine_tuning(dataset_path: str, model, tokenizer):
    """
    Performs the actual fine-tuning process using SFTTrainer.
    """
    # 1. Load the dataset
    training_status.set("Loading dataset for training...", True)
    full_dataset = Dataset.from_json(dataset_path)

    # Use the full dataset for training and disable evaluation.
    # This simplifies the training loop and avoids version-specific issues with evaluation callbacks.
    train_dataset = full_dataset
    eval_dataset = None
    print(f"✅ Using full dataset for training with {len(train_dataset)} examples. Evaluation is disabled.")

    # Explicitly set the model to training mode before passing to the trainer.
    # This is a safeguard against any prior state changes that might cause
    # the cryptic 'NoneType' error during trainer initialization.
    model.train()

    # Disable model caching to be compatible with gradient checkpointing.
    # This prevents the verbose "Caching is incompatible..." warning messages
    # by making the configuration explicit.
    if hasattr(model, "config"):
        model.config.use_cache = False

    # For compatibility with different TRL versions, attach tokenizer to the model
    # as SFTTrainer can infer it from there if it's not passed as an argument.
    model.tokenizer = tokenizer

    # 3. Configure LoRA
    training_status.set("Configuring LoRA...", True)
    lora_config = LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Explicitly specify target modules for better compatibility with various models.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 4. Configure Training Arguments
    output_dir = Path(settings.lora_path) # The final adapter will be saved here

    # Ensure the output directory exists before initializing TrainingArguments.
    # This can prevent subtle race conditions or permission errors that lead to the 'NoneType' error.
    output_dir.mkdir(parents=True, exist_ok=True)

    # For older versions of transformers, we need to calculate the number of steps
    # per epoch to configure evaluation and saving strategies correctly.
    steps_per_epoch = len(train_dataset) // (settings.training_batch_size * settings.training_grad_acc_steps)
    if steps_per_epoch == 0:
        steps_per_epoch = 1 # Ensure at least one step for very small datasets
    print(f"💡 Calculated steps per epoch: {steps_per_epoch}")


    # Choose the correct optimizer. The paged optimizer is for 4-bit quantization.
    # We check the model's `is_quantized` attribute directly for maximum reliability.
    is_quantized = getattr(model, 'is_quantized', False)
    optimizer = "paged_adamw_8bit" if is_quantized else "adamw_torch"
    print(f"💡 Using optimizer: {optimizer}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        # Explicitly set logging dir to avoid default path issues that can cause the 'NoneType' error.
        logging_dir=str(output_dir),
        # --- Core Training Loop ---
        num_train_epochs=settings.training_epochs,
        # --- Batching & Memory ---
        per_device_train_batch_size=settings.training_batch_size,
        gradient_accumulation_steps=settings.training_grad_acc_steps,
        gradient_checkpointing=True, # Saves VRAM at a small cost of speed
        # --- Optimizer & Scheduler ---
        learning_rate=settings.training_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=settings.training_warmup_ratio,
        optim=optimizer, # Use the dynamically chosen optimizer
        weight_decay=settings.training_weight_decay,
        max_grad_norm=1.0, # Add gradient clipping to prevent exploding gradients
        # --- Saving, Logging & Evaluation (using older arguments for compatibility) ---
        # Log at the end of each epoch to provide progress updates.
        logging_steps=steps_per_epoch,
        # Evaluation is disabled, so we only configure saving.
        # We still save checkpoints at the end of each epoch.
        do_eval=False,
        save_steps=steps_per_epoch,
        save_total_limit=settings.training_save_total_limit,
        report_to="none", # Disables all external reporting to prevent init errors
        # --- Hardware, Precision & Dataloader ---
        fp16=torch.cuda.is_available(),
        bf16=False, # bf16 is for Ampere GPUs, disable for broader compatibility
        dataloader_num_workers=settings.training_dataloader_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        # Explicitly tell the trainer what the label column is named to silence the warning.
        label_names=["labels"],
        # Explicitly enable the training loop.
        do_train=True,
    )

    # HACK: Manually set the save strategy. This ensures checkpoints are saved
    # correctly based on `save_steps`, even with older library versions.
    training_args.save_strategy = trainer_utils.IntervalStrategy.STEPS

    # --- Pre-flight checks and logging before initializing trainer ---
    print("\n" + "="*20 + " PRE-TRAINING DIAGNOSTICS " + "="*20)
    print(f"Model Type: {type(model)}")
    print(f"Model is in training mode: {model.training}")
    print(f"Model Device: {next(model.parameters()).device}")
    print("-" * 60)
    print(f"Tokenizer Type: {type(tokenizer)}")
    print(f"Tokenizer Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"Tokenizer EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print("-" * 60)
    print(f"Train Dataset Length: {len(train_dataset)}")
    print(f"Eval Dataset: Disabled")
    if train_dataset and len(train_dataset) > 0:
        print(f"First Training Record:\n{train_dataset[0]}")
    print("-" * 60)
    print(f"LoRA Config:\n{lora_config}")
    print("-" * 60)
    print(f"Training Arguments:\n{training_args.to_json_string()}")
    print("="*66 + "\n")

    # --- Final version check before trainer initialization ---
    trl_version = importlib.metadata.version('trl')
    print(f"💡 TRL library version check (runtime): {trl_version}")
    print("-" * 60)

    # --- Callbacks ---
    # Evaluation is disabled, so the EarlyStoppingCallback is not used.
    callbacks = []

    training_status.set("Initializing SFTTrainer...", True)

    # 5. Initialize and run the trainer
    try:
        trainer = SFTTrainer(
            model=model,
            # tokenizer=tokenizer, # Removed for broader TRL version compatibility. The tokenizer is attached to the model.
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            peft_config=lora_config,
            # Use a formatting function to extract the text from each example.
            # This is required for versions of TRL that don't support `dataset_text_field`.
            formatting_func=lambda example: example["text"],
            callbacks=callbacks,
        )

        # 6. Start training and save the model
        training_status.set("Starting fine-tuning process... (This may take a while)", True)
        print("✅ Trainer initialized successfully. Starting training...")
        trainer.train()
        trainer.save_model() # Saves to the `output_dir` defined in TrainingArguments

        # After training, the model object held by the trainer is a PeftModel.
        # We must unload the adapter here to return the persistent model object
        # (which was passed by reference) to its clean, base state.
        # This is the most reliable place to perform the cleanup.
        if hasattr(trainer.model, "unload"):
            print("💡 Unloading PEFT adapter from model post-training...")
            trainer.model.unload()
    except Exception as e:
        import traceback
        print(f"❌❌❌ CRITICAL ERROR DURING TRAINER INITIALIZATION OR TRAINING ❌❌❌")
        print(f"Error Type: {type(e)}")
        print(f"Error Message: {e}")
        print("--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------")
        # Re-raise the exception to be caught by the main run_training loop
        raise e

def _generate_and_cache_dataset(
    data_dir: Path,
    gen_model, gen_tokenizer,
    skip_behavioral_generation: bool = False
) -> Union[Path, None]:
    """
    Handles the logic for generating a synthetic dataset from source files,
    utilizing a cache to avoid re-generating data for unchanged files.
    Returns the path to the final dataset file, or None if no data was generated.
    """
    dataset_path = data_dir / "synthetic_dataset.jsonl"
    cache_dir = data_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)
    # Add .jsonl to the list of supported files to include manual datasets.
    supported_files = list(data_dir.glob('*.pdf')) + list(data_dir.glob('*.docx')) + list(data_dir.glob('*.txt')) + list(data_dir.glob('*.jsonl'))

    if not supported_files:
        training_status.set("No training files found in the 'data' directory. Nothing to do.", False)
        return None

    final_dataset = []
    # Separate files into types: pre-formatted, instructions, and general knowledge.
    preformatted_files = [f for f in supported_files if f.suffix == '.jsonl']
    instruction_files = [f for f in supported_files if "instructions" in f.name.lower() and f.suffix != '.jsonl']
    knowledge_files = [f for f in supported_files if f not in instruction_files and f not in preformatted_files]

    # Process pre-formatted JSONL files first. These are added directly to the dataset.
    for file_path in preformatted_files:
        cache_path = cache_dir / f"{file_path.name}.jsonl"
        if cache_path.exists() and cache_path.stat().st_mtime > file_path.stat().st_mtime:
            print(f"✅ Using cached data for pre-formatted file '{file_path.name}'")
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    final_dataset.append(json.loads(line))
            continue

        print(f"🧠 Reading pre-formatted data from '{file_path.name}'...")
        preformatted_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    preformatted_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"   - Warning: Skipping invalid JSON line in '{file_path.name}': {line.strip()}")
        
        final_dataset.extend(preformatted_data)
        with open(cache_path, "w", encoding="utf-8") as f:
            for item in preformatted_data:
                f.write(json.dumps(item) + "\n")

    # Process instruction and knowledge files to generate new data.
    files_to_process = knowledge_files
    if not skip_behavioral_generation:
        print("💡 Including behavioral instruction file in data generation.")
        files_to_process = instruction_files + knowledge_files
    else:
        print("💡 Skipping behavioral instruction file as requested by manual training workflow.")

    for file_path in files_to_process:
        cache_path = cache_dir / f"{file_path.name}.jsonl"
        if cache_path.exists() and cache_path.stat().st_mtime > file_path.stat().st_mtime:
            print(f"✅ Using cached data for '{file_path.name}'")
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    final_dataset.append(json.loads(line))
            continue

        print(f"🧠 Generating new data for '{file_path.name}'...")
        file_content = read_uploaded_file(str(file_path))
        generated_data = []

        if file_path in instruction_files:
            parsed_instructions = parse_instructions(file_content)
            generated_data = generate_behavioral_dataset(parsed_instructions, gen_model, gen_tokenizer)
        else:
            generated_data = generate_knowledge_dataset(file_content, gen_model, gen_tokenizer)

        final_dataset.extend(generated_data)
        with open(cache_path, "w", encoding="utf-8") as f:
            for item in generated_data:
                f.write(json.dumps(item) + "\n")

    if not final_dataset:
        return None
    
    # Save the complete dataset to the main file
    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in final_dataset:
            f.write(json.dumps(item) + "\n")
    print(f"💾 Synthetic dataset saved to '{dataset_path}'")
    return dataset_path

# --- 5. Training Process ---

def run_training(
    skip_generation: bool = False,
    skip_behavioral_generation: bool = False
): # pylint: disable=R0915, R0914
    """
    The main training pipeline. It reads ALL instruction files from the data
    directory, combines them, generates a synthetic dataset, and then kicks
    off the fine-tuning process.
    """
    print("\n" + "="*50)
    print("🏁🏁🏁 STARTING TRAINING PIPELINE 🏁🏁🏁")
    training_status.set("Starting training pipeline...", True)

    try:
        from .model_manager import model_manager
        
        data_dir = Path("data")
        dataset_path = data_dir / "synthetic_dataset.jsonl"

        if not skip_generation:
            # --- Part 1: Data Generation ---
            print("💡 Using the training base model for both data generation and fine-tuning.")
            training_status.set(f"Configuring persistent model for data generation...", True)
            gen_model, gen_tokenizer = model_manager.get_training_model()
            
            generated_dataset_path = _generate_and_cache_dataset(data_dir, gen_model, gen_tokenizer, skip_behavioral_generation=skip_behavioral_generation)
            
            if not generated_dataset_path:
                training_status.set("Could not generate any training data from the provided files. Nothing to do.", False)
                return
        else:
            print("💡 Skipping data generation. Using manually provided dataset.")
            training_status.set("Using manually provided dataset...", True)
            if not dataset_path.exists() or os.path.getsize(dataset_path) == 0:
                msg = "Dataset file not found or is empty. Nothing to train."
                print(f"❌ {msg}")
                training_status.set(f"Error: {msg}", False)
                return

        # --- Part 2: Fine-Tuning ---
        # If we skipped generation, we need to load the model now.
        # Otherwise, we can reuse the model from the generation step.
        if skip_generation:
            training_model, training_tokenizer = model_manager.get_training_model()
        else:
            training_model, training_tokenizer = gen_model, gen_tokenizer

        lora_output_path = "loras/faq-lora"
        settings.lora_path = lora_output_path # This setting is used by the chat loader to find the new adapter
        print(f"⚙️  Setting LoRA output path to: {lora_output_path}")

        perform_fine_tuning(str(dataset_path), training_model, training_tokenizer)

        training_status.set("✅ Training complete! Reloading chat model with new adapter...", True)
        print("\n" + "="*50)
        print("✅ Training complete! Reloading chat model with new adapter...")
        model_manager.get_chat_model(force_reload=True)
        
        training_status.set("✅ Model reloaded! Go to the Test Chat page to use the new version.", False)
        print("✅ Chat model reloaded successfully. Ready for testing.")
        print("="*50, flush=True)

    except Exception as e:
        import traceback
        print("❌❌❌ A CRITICAL ERROR OCCURRED IN THE TRAINING PIPELINE ❌❌❌", flush=True)
        traceback.print_exc()
        error_message = f"An error occurred during the training pipeline: {e}"
        training_status.set(f"Error: {error_message}", False)
        print("="*60, flush=True)