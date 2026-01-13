"""
FastAPI application for IndexTTS2 text-to-speech API.

This module provides a REST API interface for the IndexTTS2 text-to-speech system,
allowing programmatic access to voice synthesis with emotional expression control.
"""

import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional, List, Union
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch

# Suppress warnings for cleaner API logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment variables early
os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

# Import TTS after environment setup
from indextts.infer_v2 import IndexTTS2


class TTSRequest(BaseModel):
    """Request model for text-to-speech generation."""

    text: str = Field(..., description="Text to synthesize into speech")
    spk_audio_prompt: Optional[str] = Field(
        None, description="Path to speaker reference audio file"
    )
    emo_audio_prompt: Optional[str] = Field(
        None, description="Path to emotion reference audio file"
    )
    emo_alpha: float = Field(
        1.0, description="Emotion mixing weight (0.0-1.0)", ge=0.0, le=1.0
    )
    emo_vector: Optional[List[float]] = Field(
        None,
        description="Emotion vector as 8 floats: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]",
        min_items=8,
        max_items=8,
    )
    emo_text: Optional[str] = Field(
        None, description="Text description for emotion control"
    )
    use_random: bool = Field(False, description="Enable random sampling for emotion")
    max_text_tokens_per_segment: int = Field(
        120, description="Maximum tokens per text segment", gt=0
    )

    # Advanced generation parameters
    do_sample: bool = Field(True, description="Whether to use sampling")
    top_p: float = Field(0.8, description="Top-p sampling parameter", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(30, description="Top-k sampling parameter", ge=0)
    temperature: float = Field(0.8, description="Sampling temperature", gt=0.0)
    length_penalty: float = Field(0.0, description="Length penalty for generation")
    num_beams: int = Field(3, description="Number of beams for beam search", ge=1)
    repetition_penalty: float = Field(10.0, description="Repetition penalty", ge=0.0)
    max_mel_tokens: int = Field(
        1500, description="Maximum mel tokens to generate", gt=0
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the TTS model is loaded")
    device: str = Field(..., description="Device being used for inference")


class GenerateResponse(BaseModel):
    """Response model for speech generation."""

    audio_path: str = Field(..., description="Path to generated audio file")
    duration: float = Field(..., description="Duration of generated audio in seconds")


# Global TTS instance
tts_instance: Optional[IndexTTS2] = None


def get_tts_instance() -> IndexTTS2:
    """Get the global TTS instance, creating it if necessary."""
    global tts_instance
    if tts_instance is None:
        try:
            # Initialize TTS with default configuration
            tts_instance = IndexTTS2(
                cfg_path="checkpoints/config.yaml", model_dir="checkpoints"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TTS model: {e}")
    return tts_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    try:
        get_tts_instance()
        print("TTS model loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load TTS model on startup: {e}")

    yield

    # Shutdown
    global tts_instance
    if tts_instance is not None:
        # Clean up any resources if needed
        pass


# Create FastAPI app
app = FastAPI(
    title="IndexTTS2 API",
    description="REST API for IndexTTS2 text-to-speech synthesis with emotional expression control",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the current status of the TTS service.
    """
    try:
        tts = get_tts_instance()
        return HealthResponse(
            status="healthy", model_loaded=True, device=str(tts.device)
        )
    except Exception as e:
        return HealthResponse(status="unhealthy", model_loaded=False, device="unknown")


@app.post("/generate", response_model=GenerateResponse)
async def generate_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Generate speech from text with various control options.

    This endpoint synthesizes speech using the IndexTTS2 model with support for
    speaker cloning, emotion control, and advanced generation parameters.
    """
    try:
        tts = get_tts_instance()

        # Validate audio file paths if provided
        if request.spk_audio_prompt and not os.path.exists(request.spk_audio_prompt):
            raise HTTPException(
                status_code=400,
                detail=f"Speaker audio prompt file not found: {request.spk_audio_prompt}",
            )

        if request.emo_audio_prompt and not os.path.exists(request.emo_audio_prompt):
            raise HTTPException(
                status_code=400,
                detail=f"Emotion audio prompt file not found: {request.emo_audio_prompt}",
            )

        # Create output directory if it doesn't exist
        output_dir = Path("outputs/api")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique output path
        import time

        output_path = output_dir / f"tts_{int(time.time())}.wav"

        # Build inference arguments
        infer_kwargs = {
            "spk_audio_prompt": request.spk_audio_prompt,
            "text": request.text,
            "output_path": str(output_path),
            "emo_audio_prompt": request.emo_audio_prompt,
            "emo_alpha": request.emo_alpha,
            "emo_vector": request.emo_vector,
            "emo_text": request.emo_text,
            "use_emo_text": request.emo_text is not None,
            "use_random": request.use_random,
            "max_text_tokens_per_segment": request.max_text_tokens_per_segment,
            "do_sample": request.do_sample,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "temperature": request.temperature,
            "length_penalty": request.length_penalty,
            "num_beams": request.num_beams,
            "repetition_penalty": request.repetition_penalty,
            "max_mel_tokens": request.max_mel_tokens,
            "verbose": False,
        }

        # Generate speech
        result_path = tts.infer(**infer_kwargs)

        # Get audio duration using torchaudio
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(result_path)
            duration = waveform.shape[1] / sample_rate
        except Exception:
            duration = 0.0  # Fallback if duration calculation fails

        return GenerateResponse(audio_path=result_path, duration=duration)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Speech generation failed: {str(e)}"
        )


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file for use as a reference.

    This endpoint allows uploading audio files that can be used as speaker
    or emotion references in subsequent TTS requests.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload WAV, MP3, FLAC, or OGG files.",
            )

        # Create uploads directory
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        # Save uploaded file
        import time

        file_path = upload_dir / f"upload_{int(time.time())}_{file.filename}"

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        return {"filename": file.filename, "path": str(file_path), "size": len(content)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.get("/audio/{audio_path:path}")
async def get_audio(audio_path: str):
    """
    Retrieve a generated audio file.

    This endpoint serves audio files that were generated by the TTS system.
    """
    try:
        # Security check - only allow access to outputs and uploads directories
        allowed_dirs = ["outputs", "uploads"]
        if not any(audio_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
            raise HTTPException(status_code=403, detail="Access denied")

        file_path = Path(audio_path)

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")

        return FileResponse(
            path=file_path, media_type="audio/wav", filename=file_path.name
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve audio file: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "IndexTTS2 API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IndexTTS2 FastAPI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    uvicorn.run(
        "api:app", host=args.host, port=args.port, reload=args.reload, log_level="info"
    )
