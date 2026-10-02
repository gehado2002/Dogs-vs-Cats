"""🐶🐱 Dogs vs Cats Classifier – Streamlit app."""
import numpy as np
import streamlit as st
from PIL import Image

from src.model_utils import MODELS, available_models, load_model_cached, preprocess

st.set_page_config(page_title="Dogs vs Cats", page_icon="🐶", layout="centered")

st.title("🐶🐱 Dogs vs Cats Classifier")
st.caption("Upload a photo and let the CNN decide: cat or dog?")

# ---------------- Sidebar ----------------
models_ready = available_models()
if not models_ready:
    st.error(
        "No model files found in `models/`. Add `cnn_best_model.keras` and/or "
        "`vgg16_best_model.keras` (see README) or set the download URLs."
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Model", models_ready)
    cfg = MODELS[model_name]
    threshold = st.slider(
        "Dog threshold", 0.1, 0.9, cfg["threshold"], 0.05,
        help="Probability above which the image is classified as a dog.",
    )
    st.markdown("---")
    st.subheader("📊 Validation results")
    st.markdown(
        "| Model | Accuracy |\n|---|---|\n"
        "| Custom CNN (thr 0.4) | ~88% |\n"
        "| VGG16 fine-tuned | ~96% |"
    )

model = load_model_cached(model_name)

# ---------------- Input ----------------
tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Camera"])
with tab_upload:
    files = st.file_uploader(
        "Choose image(s)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
    )
with tab_camera:
    cam = st.camera_input("Take a picture")

images = list(files or [])
if cam is not None:
    images.append(cam)

# ---------------- Prediction ----------------
for f in images:
    img = Image.open(f)
    x = preprocess(img, cfg["size"])
    p_dog = float(model.predict(x, verbose=0)[0][0])  # sigmoid → P(dog), cat=0 dog=1
    is_dog = p_dog > threshold
    label = "Dog 🐶" if is_dog else "Cat 🐱"
    conf = p_dog if is_dog else 1 - p_dog

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, use_container_width=True)
    with col2:
        st.subheader(label)
        st.metric("Confidence", f"{conf * 100:.1f}%")
        st.progress(min(max(p_dog, 0.0), 1.0), text=f"P(dog) = {p_dog:.3f}")
    st.divider()

if not images:
    st.info("Upload an image or take a photo to get a prediction.")
