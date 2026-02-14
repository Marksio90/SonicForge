"""MusicGen HTTP Service — wraps Meta AudioCraft for SonicForge.

Runs as a standalone FastAPI service inside the musicgen Docker container.
Provides generation, status polling, and download endpoints.
"""

import asyncio
import base64
import io
import os
import uuid
from enum import Enum

import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="SonicForge MusicGen Service", version="2.0.0")

# In-memory generation store
_generations: dict[str, dict] = {}
_model = None
_model_lock = asyncio.Lock()


class GenerationStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    prompt: str
    duration: int = 30
    model_version: str = "facebook/musicgen-stereo-large"
    output_format: str = "wav"
    normalization_strategy: str = "loudness"
    bpm: int | None = None
    key: str | None = None
    continuation_audio: str | None = None  # base64-encoded WAV
    continuation_start: int | None = None


class GenerationResponse(BaseModel):
    id: str
    status: str


class StatusResponse(BaseModel):
    id: str
    status: str
    error: str | None = None


def get_model():
    """Lazy-load the MusicGen model (cached in memory)."""
    global _model
    if _model is None:
        from audiocraft.models import MusicGen

        model_version = os.environ.get(
            "MUSICGEN_MODEL_VERSION", "facebook/musicgen-stereo-large"
        )
        _model = MusicGen.get_pretrained(model_version)
    return _model


async def _run_generation(generation_id: str, request: GenerateRequest) -> None:
    """Run audio generation in background."""
    try:
        _generations[generation_id]["status"] = GenerationStatus.GENERATING

        async with _model_lock:
            model = await asyncio.to_thread(get_model)
            model.set_generation_params(duration=request.duration)

            if request.continuation_audio:
                # Audio continuation mode
                import torch
                import torchaudio

                audio_bytes = base64.b64decode(request.continuation_audio)
                audio_io = io.BytesIO(audio_bytes)
                waveform, sr = torchaudio.load(audio_io)

                if sr != model.sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, sr, model.sample_rate
                    )

                output = await asyncio.to_thread(
                    model.generate_continuation,
                    waveform.unsqueeze(0),
                    model.sample_rate,
                    [request.prompt],
                    progress=False,
                )
            else:
                output = await asyncio.to_thread(
                    model.generate, [request.prompt], progress=False
                )

        # Convert to WAV bytes
        audio_np = output[0].cpu().numpy()
        audio_io = io.BytesIO()

        # audiocraft outputs (channels, samples)
        if audio_np.ndim == 2:
            audio_np = audio_np.T  # Convert to (samples, channels) for soundfile

        sf.write(audio_io, audio_np, model.sample_rate, format="WAV")
        audio_bytes = audio_io.getvalue()

        _generations[generation_id]["status"] = GenerationStatus.COMPLETED
        _generations[generation_id]["audio"] = audio_bytes

    except Exception as e:
        _generations[generation_id]["status"] = GenerationStatus.FAILED
        _generations[generation_id]["error"] = str(e)


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerateRequest):
    """Start audio generation."""
    generation_id = str(uuid.uuid4())
    _generations[generation_id] = {
        "status": GenerationStatus.PENDING,
        "audio": None,
        "error": None,
    }

    # Run generation in background
    asyncio.create_task(_run_generation(generation_id, request))

    return GenerationResponse(id=generation_id, status="pending")


@app.get("/status/{generation_id}", response_model=StatusResponse)
async def get_status(generation_id: str):
    """Check generation status."""
    if generation_id not in _generations:
        raise HTTPException(status_code=404, detail="Generation not found")

    gen = _generations[generation_id]
    return StatusResponse(
        id=generation_id,
        status=gen["status"].value,
        error=gen.get("error"),
    )


@app.get("/download/{generation_id}")
async def download(generation_id: str):
    """Download completed audio."""
    if generation_id not in _generations:
        raise HTTPException(status_code=404, detail="Generation not found")

    gen = _generations[generation_id]
    if gen["status"] != GenerationStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Generation status: {gen['status'].value}")

    from fastapi.responses import Response

    return Response(
        content=gen["audio"],
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={generation_id}.wav"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    port = int(os.environ.get("MUSICGEN_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
