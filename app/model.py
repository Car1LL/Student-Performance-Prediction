import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
THRESHOLD_PATH = ARTIFACTS_DIR / "best_threshold.json"

MINOR_CLASS = 0

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, 'r') as f:
    threshold = json.load(f)["threshold"]