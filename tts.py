import torch
import soundfile as sf
from fastapi import FastAPI
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from io import BytesIO

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

fastpitch = FastPitchModel.from_pretrained(
    model_name="tts_en_fastpitch"
).to(device)

hifigan = HifiGanModel.from_pretrained(
    model_name="tts_hifigan"
).to(device)

@app.post("/speak")
def speak(text: str):
    with torch.no_grad():
        tokens = fastpitch.parse(text)
        spec = fastpitch.generate_spectrogram(
            tokens=tokens,
            pace=0.9,
            pitch=0.95
        )
        audio = hifigan.convert_spectrogram_to_audio(spec)

    audio_np = audio.cpu().numpy()[0]

    buffer = BytesIO()
    sf.write(buffer, audio_np, samplerate=22050, format="WAV")
    buffer.seek(0)

    return buffer.read()
