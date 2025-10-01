#!/usr/bin/env python3
"""
File parsing utilities for the chatbot
"""

import os
import PyPDF2
import docx
from typing import Optional, List, Dict
import re
import json # ⬅️ Corrected: Added the missing import for the json module.

class FileParser:
    """Parse various file formats to extract text content"""
    
    @staticmethod
    def parse_txt(file_path: str) -> Optional[str]:
        """Extract text from a TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading TXT file: {e}")
            return None
    
    @staticmethod
    def parse_pdf(file_path: str) -> Optional[str]:
        """Extract text from a PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return None
    
    @staticmethod
    def parse_docx(file_path: str) -> Optional[str]:
        """Extract text from a DOCX file"""
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"Error reading DOCX file: {e}")
            return None
    
    @staticmethod
    def parse_file(file_path: str) -> Optional[str]:
        """Parse any supported file format"""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            return FileParser.parse_txt(file_path)
        elif ext == '.pdf':
            return FileParser.parse_pdf(file_path)
        elif ext == '.docx':  # ⬅️ Corrected: Removed '.doc' as python-docx can't handle it.
            return FileParser.parse_docx(file_path)
        else:
            print(f"Unsupported file format: {ext}")
            return None
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction by frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Remove common words
        common_words = {'the', 'and', 'for', 'with', 'this', 'that', 'are', 'from', 'has', 'have', 
                        'was', 'were', 'will', 'would', 'should', 'could', 'their', 'there', 'which'}
        
        keywords = [(word, count) for word, count in word_freq.items() 
                    if word not in common_words and count > 1]
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, count in keywords[:max_keywords]]
    
    @staticmethod
    def prepare_training_data(text: str, output_path: str, max_examples: int = 100):
        """Prepare text data for fine-tuning"""
        chunks = FileParser.chunk_text(text, 500)  # Smaller chunks for training
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks[:max_examples]):
                # Create a simple instruction-following format
                training_example = {
                    "instruction": "Answer the following question or respond to the statement in a natural, human-like way.",
                    "input": chunk,
                    "output": ""  # This would be filled with human-like responses
                }
                f.write(json.dumps(training_example) + "\n")
        
        print(f"Prepared {min(len(chunks), max_examples)} training examples in {output_path}")