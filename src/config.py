"""Shared application configuration; emotion names are future placeholders."""
from pathlib import Path

APP_TITLE = "Student Feedback Emotion Analysis"
APP_SUBTITLE = "Discover the emotions behind student feedback using Natural Language Processing."
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATASET_PATH = PROJECT_ROOT / "data" / "sample_student_feedback.csv"
REQUIRED_DATASET_COLUMNS = (
    "feedback_id", "feedback", "course", "subject", "semester", "rating", "feedback_date"
)
NAVIGATION_LABELS = ("Overview", "Analyze Feedback", "Emotion Dashboard", "About")
FUTURE_EMOTION_CATEGORIES = (
    "joy", "sadness", "anger", "fear", "surprise", "frustration", "satisfaction", "neutral"
)
