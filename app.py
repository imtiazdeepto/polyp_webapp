"""Streamlit frontend for the polyp segmentation CAD system."""

import base64
import os
from io import BytesIO

import numpy as np
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Polyp Segmentation CAD", layout="wide")

USE_LOCAL_MODEL = os.environ.get("USE_LOCAL_MODEL", "0").lower() in {"1", "true", "yes"}


def _resolve_api_url() -> str:
    backend_host = os.environ.get("BACKEND_HOST", "").strip()
    if backend_host:
        host = backend_host.removeprefix("https://").removeprefix("http://").rstrip("/")
        return f"https://{host}/predict"
    return os.environ.get("API_URL", "http://localhost:8000/predict")


API_URL = _resolve_api_url()

st.title("Polyp Segmentation CAD")
st.caption("UNet + ResNet-18 based colonoscopy polyp detection and segmentation")

if USE_LOCAL_MODEL:
    st.info("Running in local inference mode (model loaded inside this app).")

ALLOWED_TYPES = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE_BYTES = int(os.environ.get("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024


def decode_b64(b64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64_str)))


def reset_session() -> None:
    st.session_state.pop("uploaded_bytes", None)
    st.session_state.pop("uploaded_name", None)
    st.session_state.pop("uploaded_type", None)
    st.session_state.pop("last_result", None)


@st.cache_resource(show_spinner="Loading model...")
def _load_local_model():
    from inference import load_model

    return load_model()


def analyze_image(file_bytes: bytes) -> dict | None:
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        st.error(f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.")
        return None

    if USE_LOCAL_MODEL:
        try:
            _load_local_model()
            from inference import run_inference

            image_np = np.array(Image.open(BytesIO(file_bytes)).convert("RGB"))
            return run_inference(image_np)
        except Exception as exc:
            st.error(f"Local inference failed: {exc}")
            return None

    try:
        response = requests.post(
            API_URL,
            files={
                "file": (
                    st.session_state.uploaded_name,
                    file_bytes,
                    st.session_state.uploaded_type,
                )
            },
            timeout=120,
        )
    except requests.ConnectionError:
        st.error(
            "Could not connect to the backend. "
            "Make sure the FastAPI server is running and API_URL is correct."
        )
        return None
    except requests.Timeout:
        st.error("The request timed out. The backend may be busy or unreachable.")
        return None
    except requests.RequestException:
        st.error("Network error while contacting the backend. Please try again.")
        return None

    if response.status_code == 200:
        return response.json()

    detail = "Unknown error"
    try:
        payload = response.json()
        detail = payload.get("detail", detail)
    except ValueError:
        detail = response.text or detail
    st.error(f"Analysis failed ({response.status_code}): {detail}")
    return None


if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
    st.session_state.uploaded_name = None
    st.session_state.uploaded_type = None
    st.session_state.last_result = None

uploaded_file = st.file_uploader(
    "Upload a colonoscopy image",
    type=ALLOWED_TYPES,
    help="Supported formats: JPG, JPEG, PNG (max 10 MB)",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    if file_bytes != st.session_state.uploaded_bytes:
        st.session_state.uploaded_bytes = file_bytes
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.uploaded_type = uploaded_file.type or "image/png"
        st.session_state.last_result = None

if st.session_state.uploaded_bytes:
    preview = Image.open(BytesIO(st.session_state.uploaded_bytes))
    st.image(preview, caption="Uploaded Image", width=320)

    analyze_col, clear_col = st.columns([1, 4])
    with analyze_col:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    with clear_col:
        if st.button("Clear", use_container_width=False):
            reset_session()
            st.rerun()

    if analyze_clicked:
        with st.spinner("Model inference in progress..."):
            result = analyze_image(st.session_state.uploaded_bytes)
            if result is not None:
                st.session_state.last_result = result

if st.session_state.last_result:
    result = st.session_state.last_result

    st.subheader("Diagnostic Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Polyp Detected", "Yes" if result["tumor_present"] else "No")
    col2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
    col3.metric("Mask Area (px)", result["mask_area_px"])

    st.subheader("Visual Results")
    img1, img2, img3, img4, img5 = st.columns(5)
    img1.image(decode_b64(result["original_image"]), caption="Original", use_container_width=True)
    img2.image(decode_b64(result["mask_image"]), caption="Mask", use_container_width=True)
    img3.image(decode_b64(result["overlay_image"]), caption="Overlay", use_container_width=True)
    img4.image(decode_b64(result["contour_image"]), caption="Contours", use_container_width=True)
    img5.image(decode_b64(result["bbox_image"]), caption="Bounding Boxes", use_container_width=True)

    if result.get("bboxes"):
        with st.expander("Bounding box coordinates (JSON)"):
            st.json(result["bboxes"])
