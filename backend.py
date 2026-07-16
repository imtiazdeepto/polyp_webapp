"""FastAPI backend for polyp segmentation inference."""

import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from inference import load_model, run_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("polyp_backend")

MAX_FILE_SIZE_BYTES = int(os.environ.get("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as exc:
        logger.exception("Failed to load model at startup")
        raise RuntimeError(f"Model load failed: {exc}") from exc
    yield


app = FastAPI(title="Polyp Segmentation CAD API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %s (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "%s %s failed after %.1f ms",
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise


def _validate_upload(file: UploadFile) -> None:
    filename = (file.filename or "").lower()
    extension = os.path.splitext(filename)[1]

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{content_type}'. Upload a JPG or PNG image.",
        )


@app.get("/")
def health_check():
    return {"status": "Polyp Segmentation API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    _validate_upload(file)

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    try:
        pil_img = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(pil_img)
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("Image decode failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Please upload a valid JPG or PNG file.",
        ) from exc

    try:
        return run_inference(image_np)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed. Please try again.") from exc
