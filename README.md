# PoSTAA
PoSTAA (Positive Self Talk AI Assistant). A highly personalized experience with one's own voice and own avatar to anchor a growth mindset.
It uses Nvidia's open source technology NeMo Agentic Toolkit for Generative AI. RAG is being utilized to retrieve information and for Nvidia Riva's text to speech (TTS) service is selected for modeling. 


ARCHITECTURE:
Text Generator (LLM / rules)
        ↓
NeMo TTS (FastPitch / Magpie)
        ↓
HiFiGAN Vocoder
        ↓
Riva (real-time serving)
        ↓
Mobile / Web / Wearable App


BACKEND API DEPLOYMENT:
pip install fastapi uvicorn nemo_toolkit[tts] soundfile torch

RUN BACKEND:
uvicorn tts_service:app --host 0.0.0.0 --port 8000

