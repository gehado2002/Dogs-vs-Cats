import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent

# Class indices from flow_from_directory (alphabetical): cat=0, dog=1
MODELS = {
    "VGG16 (fine-tuned)": {
        "path": ROOT / "models" / "vgg16_best_model.keras",
        "size": (224, 224),
        "threshold": 0.5,
        "url_env": "VGG_MODEL_URL",
    },
    "Custom CNN": {
        "path": ROOT / "models" / "cnn_best_model.keras",
        "size": (150, 150),
        "threshold": 0.4,  # tuned in the notebook
        "url_env": "CNN_MODEL_URL",
    },
}


def _get_url(env_key: str):
    """Read a download URL from env var or Streamlit secrets."""
    if os.getenv(env_key):
        return os.getenv(env_key)
    try:
        return st.secrets.get(env_key)
    except Exception:
        return None


def _download_if_missing(cfg: dict) -> bool:
    path: Path = cfg["path"]
    if path.exists():
        return True
    url = _get_url(cfg["url_env"])
    if not url:
        return False
    import gdown

    path.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner(f"Downloading {path.name} ..."):
        gdown.download(url, str(path), quiet=True, fuzzy=True)
    return path.exists()


def available_models() -> list:
    return [name for name, cfg in MODELS.items() if _download_if_missing(cfg)]


@st.cache_resource(show_spinner="Loading model...")
def load_model_cached(name: str):
    from tensorflow.keras.models import load_model

    return load_model(MODELS[name]["path"], compile=False)


def preprocess(img: Image.Image, size: tuple) -> np.ndarray:
    """Same pipeline as training: RGB -> resize (nearest, Keras default) -> /255."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize(size, Image.NEAREST)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)
