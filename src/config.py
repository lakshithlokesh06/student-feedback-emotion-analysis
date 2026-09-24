"""Shared application configuration; emotion names are future placeholders."""
from pathlib import Path

APP_TITLE = "Student Feedback Emotion Analysis"
APP_SUBTITLE = "Discover the emotions behind student feedback using Natural Language Processing."
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATASET_PATH = PROJECT_ROOT / "data" / "sample_student_feedback.csv"
REQUIRED_DATASET_COLUMNS = (
    "feedback_id", "feedback", "course", "subject", "semester", "rating", "feedback_date"
)
NAVIGATION_LABELS = ("Overview", "Analyze Feedback", "Emotion Dashboard", "Model Evaluation", "About")
FUTURE_EMOTION_CATEGORIES = (
    "joy", "sadness", "anger", "fear", "surprise", "frustration", "satisfaction", "neutral"
)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 100
MIN_FEEDBACK_LENGTH = 3
PREVIEW_ROWS = 50
CONTEXT_COLUMNS = ("course", "subject", "semester", "rating", "feedback_date")

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
MODEL_REVISION = "0e1cd914e3d46199ed785853e12b57304e04178b"
MODEL_CACHE_DIR = PROJECT_ROOT / ".model_cache"
MODEL_EMOTIONS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
INFERENCE_BATCH_SIZE = 8
MAX_MODEL_TOKENS = 512
LOW_CONFIDENCE_THRESHOLD = 0.50

LABELED_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_labeled_feedback.csv"
EVALUATION_CONFIDENCE_EDGES = (0.0, 0.50, 0.70, 0.85, 1.0)
HIGH_CONFIDENCE_THRESHOLD = 0.85
AMBIGUITY_MARGIN_THRESHOLD = 0.10
CALIBRATION_BINS = 10
