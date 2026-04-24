from io import BytesIO
from typing import Literal, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from kokoro import KPipeline


app = FastAPI(title="Open Source TTS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# English pipeline
pipeline = KPipeline(lang_code="a")

ALLOWED_VOICES = {
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
}


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)
    voice: str = "af_heart"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    format: Literal["wav"] = "wav"


@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Python TTS API is running",
        "route": "/api/text-to-speech",
    }


@app.get("/api/voices")
def list_voices():
    return {
        "voices": sorted(ALLOWED_VOICES)
    }


@app.post("/api/text-to-speech")
def text_to_speech(payload: TTSRequest):
    text = payload.text.strip()
    voice = payload.voice.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    if voice not in ALLOWED_VOICES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid voice.",
                "allowed_voices": sorted(ALLOWED_VOICES),
            },
        )

    try:
        audio_parts = []

        # Kokoro generator returns chunks
        generator = pipeline(text, voice=voice, speed=payload.speed)

        sample_rate: Optional[int] = None

        for _, _, audio in generator:
            if audio is None:
                continue

            if sample_rate is None:
                sample_rate = 24000  # Kokoro commonly outputs 24kHz

            audio_parts.append(np.asarray(audio, dtype=np.float32))

        if not audio_parts:
            raise HTTPException(status_code=500, detail="No audio generated.")

        full_audio = np.concatenate(audio_parts, axis=0)

        buffer = BytesIO()
        sf.write(buffer, full_audio, samplerate=sample_rate or 24000, format="WAV")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="speech.wav"',
                "Cache-Control": "no-store",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.exception_handler(Exception)
def global_exception_handler(_, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Unexpected server error", "details": str(exc)},
    )