#!/usr/bin/env python3
"""
Enhanced Humanized Mistral 7B GGUF Chatbot with File Parsing and Fine-Tuning Support
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from llama_cpp import Llama
import readline  # For better input handling

# Import our file parser
from file_parser import FileParser

class HumanizedChatbot:
    """Enhanced Mistral 7B GGUF Chatbot with humanization techniques"""
    
    def __init__(self, model_path: str, lora_path: Optional[str] = None, n_gpu_layers: int = 0, n_ctx: int = 4096):
        """Initialize the chatbot with a GGUF model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        
        # Model parameters for more human-like responses
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            chat_format="mistral-instruct",
            verbose=False,
            # Parameters for more human-like responses
            temperature=0.8,        # Higher temperature for more creativity
            top_k=40,               # Consider more tokens
            top_p=0.9,              # Nucleus sampling for diversity
            repeat_penalty=1.1,     # Reduce repetition
        )
        
        self.history = []
        print("Model loaded successfully!")
        
        # Enhanced humanization settings
        self.temperature = 0.8
        self.top_p = 0.9
        self.max_tokens = 512
        
        # Load humanization prompts
        self.humanization_prompts = self.load_humanization_prompts()
        
        # System prompt for human-like responses
        self.system_prompt = """You are a friendly, helpful, and professional AI assistant. 
Respond in a natural, conversational, and engaging manner. Be concise, polite, and empathetic. 
Avoid technical jargon unless necessary. Use contractions like "you're" and "don't" to sound more human.
If you don't know something, say so rather than making up information."""
        
        # File context
        self.file_context = ""
        
        # Initialize file parser
        self.file_parser = FileParser()
        
        # Conversation statistics for adaptive humanization
        self.conversation_stats = {
            "total_exchanges": 0,
            "avg_response_length": 0,
            "formality_level": 0.5  # 0.0 = casual, 1.0 = formal
        }
    
    def load_humanization_prompts(self) -> List[str]:
        """Load prompts that encourage human-like responses"""
        prompts = [
            "Respond as if you're having a friendly conversation with a colleague.",
            "Answer in a natural, conversational tone as if speaking to a friend.",
            "Keep your response concise, friendly, and approachable.",
            "Use everyday language that's easy to understand.",
            "Add a touch of warmth and personality to your response.",
            "Imagine you're explaining this to someone who's new to the topic.",
            "Respond with the same tone you'd use in a casual professional setting.",
            "Make your answer engaging and interesting to read.",
            "Use examples or analogies to make your point clearer.",
            "Show empathy and understanding in your response."
        ]
        return prompts
    
    def load_file_context(self, file_path: str) -> bool:
        """Load and parse a file to use as context"""
        print(f"Loading file: {file_path}")
        content = self.file_parser.parse_file(file_path)
        
        if content:
            # Extract keywords for better context understanding
            keywords = self.file_parser.extract_keywords(content)
            print(f"Extracted keywords: {', '.join(keywords[:5])}...")
            
            # Create a summary of the file content for context
            chunks = self.file_parser.chunk_text(content, 2000)
            self.file_context = chunks[0]  # Use first chunk as context
            
            if len(chunks) > 1:
                print(f"Note: File is large ({len(content)} characters). Using first 2000 characters as context.")
            
            print("File content loaded as context!")
            return True
        else:
            print("Failed to parse file!")
            return False
    
    def enhance_humanization(self, response: str) -> str:
        """Apply multiple techniques to make responses more human-like"""
        # Remove common robotic phrases
        robotic_phrases = [
            "As an AI language model",
            "According to the provided context",
            "Based on my training data",
            "I am an AI",
            "As an artificial intelligence",
            "I don't have personal opinions",
            "I don't have personal experiences",
            "My purpose is to",
            "I was designed to",
            "My function is to"
        ]
        
        for phrase in robotic_phrases:
            response = response.replace(phrase, "")
        
        # Add conversational markers
        conversational_starters = [
            "You know, ",
            "Actually, ",
            "I think ",
            "In my experience, ",
            "From what I understand, ",
            "The way I see it, ",
            "Honestly, ",
            "Interestingly, "
        ]
        
        # Occasionally add a conversational starter
        import random
        if random.random() < 0.3 and not any(response.startswith(starter) for starter in conversational_starters):
            response = random.choice(conversational_starters) + response.lower()
        
        # Ensure proper formatting
        response = response.strip()
        
        # Capitalize first letter if needed
        if response and not response[0].isupper():
            response = response.capitalize()
            
        # Remove extra whitespace
        import re
        response = re.sub(r'\s+', ' ', response)
        
        # Add occasional conversational fillers (sparingly)
        conversational_fillers = ["You know, ", "I mean, ", "Well, ", "So, "]
        if random.random() < 0.2:
            response = random.choice(conversational_fillers) + response
        
        # Update conversation statistics
        self.update_conversation_stats(response)
        
        # Adjust formality based on conversation history
        response = self.adjust_formality(response)
        
        return response
    
    def update_conversation_stats(self, response: str):
        """Update conversation statistics for adaptive humanization"""
        self.conversation_stats["total_exchanges"] += 1
        
        # Update average response length
        current_avg = self.conversation_stats["avg_response_length"]
        new_length = len(response.split())
        self.conversation_stats["avg_response_length"] = (
            (current_avg * (self.conversation_stats["total_exchanges"] - 1) + new_length) / 
            self.conversation_stats["total_exchanges"]
        )
        
        # Adjust formality based on response length and content
        if new_length < 10:
            self.conversation_stats["formality_level"] = max(0.0, self.conversation_stats["formality_level"] - 0.1)
        elif new_length > 30:
            self.conversation_stats["formality_level"] = min(1.0, self.conversation_stats["formality_level"] + 0.1)
    
    def adjust_formality(self, response: str) -> str:
        """Adjust the formality level of the response based on conversation history"""
        formality = self.conversation_stats["formality_level"]
        
        # More casual
        if formality < 0.3:
            # Use more contractions and casual language
            response = response.replace("it is", "it's")
            response = response.replace("that is", "that's")
            response = response.replace("what is", "what's")
            response = response.replace("who is", "who's")
            response = response.replace("where is", "where's")
            response = response.replace("how is", "how's")
            response = response.replace("why is", "why's")
            response = response.replace("when is", "when's")
        
        # More formal
        elif formality > 0.7:
            # Use fewer contractions and more formal language
            response = response.replace("it's", "it is")
            response = response.replace("that's", "that is")
            response = response.replace("what's", "what is")
            response = response.replace("who's", "who is")
            response = response.replace("where's", "where is")
            response = response.replace("how's", "how is")
            response = response.replace("why's", "why is")
            response = response.replace("when's", "when is")
        
        return response
    
    def generate_response(self, message: str) -> str:
        """Generate a response to the user's message"""
        # Build the conversation history
        messages = []
        
        # Add system prompt with file context and humanization guidance
        full_system_prompt = self.system_prompt
        
        # Add humanization prompt (rotate through different prompts)
        import random
        humanization_prompt = random.choice(self.humanization_prompts)
        full_system_prompt += f"\n\n{humanization_prompt}"
        
        if self.file_context:
            full_system_prompt += f"\n\nUse the following information to inform your responses:\n{self.file_context}"
            
        messages.append({"role": "system", "content": full_system_prompt})
        
        # Add conversation history
        messages.extend(self.history)
        
        # Add the current message
        messages.append({"role": "user", "content": message})
        
        try:
            # Generate response
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=["</s>", "###", "[INST]"],
                stream=False
            )
            
            # Extract the response text
            response_text = response['choices'][0]['message']['content']
            
            # Apply humanization techniques
            humanized_response = self.enhance_humanization(response_text)
            
            return humanized_response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat(self, message: str) -> str:
        """Process a chat message and return response"""
        response = self.generate_response(message)
        
        # Update history (keeping a reasonable limit)
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        
        # Keep only the last 10 exchanges to manage context length
        if len(self.history) > 20:
            self.history = self.history[-20:]
            
        return response
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n" + "="*60)
        print("Enhanced Humanized Mistral 7B Chatbot")
        print("Commands:")
        print("  /load <file> - Load a file as context")
        print("  /clear      - Clear conversation history")
        print("  /context    - Show current context")
        print("  /stats      - Show conversation statistics")
        print("  /quit       - Exit the chatbot")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == '/quit':
                    break
                elif user_input.lower() == '/clear':
                    self.history = []
                    self.conversation_stats = {
                        "total_exchanges": 0,
                        "avg_response_length": 0,
                        "formality_level": 0.5
                    }
                    print("Conversation history and statistics cleared.")
                    continue
                elif user_input.lower() == '/context':
                    if self.file_context:
                        print("Current context:")
                        print(self.file_context[:500] + "..." if len(self.file_context) > 500 else self.file_context)
                    else:
                        print("No file context loaded.")
                    continue
                elif user_input.lower() == '/stats':
                    print("Conversation Statistics:")
                    print(f"  Total exchanges: {self.conversation_stats['total_exchanges']}")
                    print(f"  Average response length: {self.conversation_stats['avg_response_length']:.1f} words")
                    print(f"  Formality level: {self.conversation_stats['formality_level']:.2f} (0=casual, 1=formal)")
                    continue
                elif user_input.startswith('/load '):
                    file_path = user_input[6:].strip()
                    if file_path:
                        self.load_file_context(file_path)
                    else:
                        print("Please specify a file path.")
                    continue
                elif not user_input:
                    continue
                    
                print("Bot: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description="Enhanced Humanized Mistral 7B GGUF Chatbot")
    parser.add_argument("--model-path", required=True, help="Path to the GGUF model file")
    parser.add_argument("--lora-path", help="Path to a LoRA adapter (not yet supported with GGUF)")
    parser.add_argument("--gpu-layers", type=int, default=0, help="Number of layers to offload to GPU (0 for CPU-only)")
    parser.add_argument("--context-size", type=int, default=4096, help="Context window size")
    parser.add_argument("--file", help="Path to a file to use as context (txt, pdf, docx)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please download a model first using: python model_downloader.py")
        sys.exit(1)
    
    try:
        # Initialize chatbot
        chatbot = HumanizedChatbot(
            model_path=args.model_path,
            lora_path=args.lora_path,
            n_gpu_layers=args.gpu_layers,
            n_ctx=args.context_size
        )
        
        # Load file if provided
        if args.file:
            chatbot.load_file_context(args.file)
        
        # Start interactive chat
        chatbot.interactive_chat()
        
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()