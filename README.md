# Churn Radar – Customer Churn Prediction

FastAPI + static frontend, ready for Vercel.

- `main.py` – API (`POST /api/predict`, `GET /api/meta`, `GET /api/health`) and local static serving
- `public/` – frontend (`index.html`, `style.css`, `script.js`)
- `artifacts/` – deployed model: LightGBM trees (`trees.json`) + preprocessing (`preprocess.json`)
- `tools/` – training-side files and `build_artifacts.py` (regenerates `artifacts/`; not deployed)

## Run locally
```bash
python -m venv .venv
source .venv/Scripts/activate     # Git Bash on Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m uvicorn main:app --reload
```
Open http://127.0.0.1:8000 (API docs: /docs).

## Deploy
Push to GitHub, import the repo in Vercel, press Deploy (no settings needed).

## Regenerate artifacts (only if you retrain the model)
```bash
pip install lightgbm scikit-learn pandas numpy joblib
python tools/build_artifacts.py
```
