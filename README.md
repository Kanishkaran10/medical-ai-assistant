# medical-ai-assistant
![GitHub stars](https://img.shields.io/github/stars/Kanishkaran10/medical-ai-assistant?style=social)
![GitHub forks](https://img.shields.io/github/forks/Kanishkaran10/medical-ai-assistant?style=social)
![GitHub license](https://img.shields.io/github/license/Kanishkaran10/medical-ai-assistant)
![GitHub issues](https://img.shields.io/github/issues/Kanishkaran10/medical-ai-assistant)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

🩺 Medical AI Assistant (Multilingual + Voice Enabled)
An advanced Medical Multilingual Chatbot with Voice Support built using Streamlit, NLP models, and speech processing.
This assistant can understand medical queries in multiple languages, translate them, generate AI-based responses, and even speak answers back to the user.

🚀 Features
Voice Input Support
Uses speech_recognition to capture user speech
Text-to-Speech Output
Converts responses to audio using gTTS
Multilingual Support

Detects language using langdetect
Translates using deep-translator (Google Translate)
Fallback to MarianMT models if translation fails
Medical NLP Model
Uses fine-tuned model: kaniskaran/medical-finetuned
Medical Query Detection
Filters non-medical queries using keyword matching
Chat Interface
Interactive UI with Streamlit chat system
Smart Response Generation
Generates detailed answers using HuggingFace Transformers
Auto-expands short responses for better clarity
Text Preprocessing
Cleans and normalizes user input

Tech Stack
Frontend: Streamlit
Backend: Python
AI/NLP:
Hugging Face Transformers
Torch
Speech Processing:
SpeechRecognition
gTTS
Translation:
deep-translator (GoogleTranslator)
MarianMT (Helsinki-NLP models)
Utilities:
langdetect
regex
logging

📁 Project Structure
Medical-AI-Assistant/
│── app.py                 
│── preprocessing.py       
│── config.py              
│── requirements.txt        
│── README.md  

Installation
Clone the repository
git clone https://github.com/Kanishkaran10/medical-ai-assistant.git

cd medical-ai-assistant

Create virtual environment
python -m venv venv
venv\Scripts\activate  
Install dependencies
pip install -r requirements.txt
Run the App

streamlit run app.py

Open in browser:
http://localhost:8501

Pipeline
User enters text or speaks (voice mode)
Speech is converted → text (speech_recognition)
Language is detected (langdetect)
Input is translated → English (deep-translator / MarianMT)
Query is validated (medical or not)
Prompt is generated for AI model
Model generates response (transformers)
Response is:
Displayed as text
Converted to speech (gTTS)
Core Components
Model Loading
Loads tokenizer & model using HuggingFace
Supports GPU (cuda) if available
Translation System
Primary: GoogleTranslator
Fallback: MarianMT models (cached for performance)
Medical Query Filter
Uses a large keyword set to ensure domain relevance
Response Generator
Uses controlled generation:
Low temperature (0.3)
Top-p sampling
Repetition penalty
Auto-extends short answers for better detail
 UI Features
 
💬 Chat-based interface

🎤 Toggle Voice Mode

🗑️ Clear chat history

🔊 Audio playback for responses

Disclaimer

This application is for educational purposes only.
It does NOT provide medical diagnosis or treatment.
Always consult a licensed medical professional.

Future Improvements
Add more regional language support (Tamil, Hindi, etc.)
Improve medical model accuracy
Mobile optimization
Deploy on cloud (AWS / HuggingFace Spaces)
Integrate real medical APIs
Improve voice accuracy & noise handling
Contributing

Contributions are welcome!

fork → clone → create branch → commit → push → pull request
📜 License

MIT License

Author
Kanishkaran SJ
