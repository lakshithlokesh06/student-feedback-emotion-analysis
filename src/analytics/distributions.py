import pandas as pd
from src.config import MODEL_EMOTIONS
from src.analytics.metrics import analyzed


def emotion_distribution(data: pd.DataFrame) -> pd.DataFrame:
    rows = analyzed(data)
    counts = rows.emotion_label.value_counts().reindex(MODEL_EMOTIONS, fill_value=0)
    result = counts.rename_axis('Emotion').reset_index(name='Count')
    result['Percentage'] = 100 * result.Count / len(rows) if len(rows) else 0.0
    return result


def confidence_by_emotion(data: pd.DataFrame) -> pd.DataFrame:
    return analyzed(data).groupby('emotion_label').emotion_confidence.agg(['count', 'mean', 'median']).reindex(MODEL_EMOTIONS).dropna(subset=['mean']).reset_index()
