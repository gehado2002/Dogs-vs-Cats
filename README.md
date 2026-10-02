# 🐶🐱 Dogs vs Cats Classifier

Streamlit app for classifying cat/dog images using two Keras models trained on the
[Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset (25,000 images, 80/20 split).

| Model | Input | Val. accuracy |
|---|---|---|
| Custom CNN (4 conv blocks + BN + GAP) | 150×150 | ~88% (threshold 0.4) |
| VGG16 transfer learning + fine-tuning | 224×224 | ~96% |

## Project structure
```
dogs-vs-cats-classifier/
├── app.py                  # Streamlit UI
├── src/model_utils.py      # model loading + preprocessing
├── models/                 # .keras files (Git LFS)
├── notebooks/              # Colab training notebook
├── assets/                 # screenshots
├── .streamlit/config.toml
├── requirements.txt
├── .gitattributes          # Git LFS rules
└── .github/workflows/ci.yml
```

## Run locally
```bash
git clone https://github.com/<your-username>/dogs-vs-cats-classifier.git
cd dogs-vs-cats-classifier
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Models
Copy `cnn_best_model.keras` and `vgg16_best_model.keras` into `models/`.
Files over 100 MB need [Git LFS](https://git-lfs.com):
```bash
git lfs install
git lfs track "*.keras"
```
Alternative: upload models to Google Drive (share "Anyone with the link") and set
`CNN_MODEL_URL` / `VGG_MODEL_URL` in `.streamlit/secrets.toml` or as env vars; the app downloads them on first run.

## Deploy (Streamlit Community Cloud)
1. Push the repo to GitHub.
2. share.streamlit.io → New app → select repo, branch `main`, file `app.py`.
3. (If models are on Drive) add the URLs under *Advanced settings → Secrets*:
```toml
CNN_MODEL_URL = "https://drive.google.com/file/d/XXXX/view"
VGG_MODEL_URL = "https://drive.google.com/file/d/YYYY/view"
```

## Training notes
- Class mapping: `cat = 0`, `dog = 1`; sigmoid output = P(dog)
- Images rescaled by 1/255 (no VGG `preprocess_input`, matching training)
- VGG16: 10 epochs frozen (lr 1e-4) + 5 epochs fine-tuning last 4 layers (lr 1e-5)
