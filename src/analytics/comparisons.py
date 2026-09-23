import numpy as np
import pandas as pd
from src.config import MODEL_EMOTIONS
from src.analytics.metrics import analyzed, dominant


def group_comparison(data: pd.DataFrame, column: str, limit: int = 15) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if column not in data:
        return pd.DataFrame(), pd.DataFrame(), 0
    rows = analyzed(data).dropna(subset=[column])
    counts = rows[column].value_counts()
    # Ties are deterministic by string label; groups are not ranked by emotion.
    groups = sorted(counts.index, key=lambda value: (-counts[value], str(value)))[:limit]
    summaries, composition = [], []
    for group in groups:
        part = rows.loc[rows[column].eq(group)]
        summaries.append({'Group': group, 'Count': len(part), 'Dominant emotion': dominant(part),
                          'Average confidence': part.emotion_confidence.mean()})
        emotions = part.emotion_label.value_counts()
        for emotion in MODEL_EMOTIONS:
            count = int(emotions.get(emotion, 0))
            composition.append({'Group': group, 'Emotion': emotion, 'Count': count, 'Percentage': 100 * count / len(part)})
    return pd.DataFrame(summaries), pd.DataFrame(composition), len(counts)


def rating_comparison(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rows = analyzed(data)
    if 'rating' not in rows:
        return pd.DataFrame(), pd.DataFrame(), len(rows)
    ratings = pd.to_numeric(rows.rating, errors='coerce')
    valid = ratings.notna() & np.isfinite(ratings)
    rows = rows.loc[valid].assign(rating=ratings.loc[valid])
    averages = rows.groupby('emotion_label').rating.agg(['mean', 'count']).reindex(MODEL_EMOTIONS).dropna(subset=['mean']).reset_index()
    distribution = rows.groupby(['rating', 'emotion_label']).size().reset_index(name='Count')
    distribution['Percentage'] = distribution.Count / distribution.groupby('rating').Count.transform('sum') * 100
    return averages, distribution, int((~valid).sum())
