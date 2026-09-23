"""Canonical analytics view and shared descriptive metrics."""
import pandas as pd
from src.config import MODEL_EMOTIONS


def analytics_view(result, preparation_fields: dict, contexts: dict) -> pd.DataFrame:
    """Narrow copy with positional index for lossless selection of original rows."""
    view = pd.DataFrame(index=range(len(result.data)))
    for role, column in result.fields.items():
        view[role] = result.data[column].to_numpy()
    for role in ('course', 'subject', 'semester'):
        column = contexts.get(role)
        if column in result.data:
            values = result.data[column].astype('string').str.strip().replace('', pd.NA)
            view[role] = values.to_numpy()
    for role, field in (('rating', 'rating_numeric'), ('date', 'feedback_date_parsed')):
        column = preparation_fields.get(field)
        if column in result.data:
            view[role] = result.data[column].to_numpy()
    return view


def analyzed(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[data['emotion_status'].eq('analyzed')]


def dominant(data: pd.DataFrame) -> str:
    counts = analyzed(data)['emotion_label'].value_counts()
    return ', '.join(label for label in MODEL_EMOTIONS if counts.get(label, 0) == counts.max()) if len(counts) else 'Not yet analyzed'


def metrics(data: pd.DataFrame, threshold: float) -> dict:
    rows = analyzed(data)
    scores = rows['emotion_confidence']
    low = int(scores.lt(threshold).sum())
    return dict(total=len(data), analyzed=len(rows), dominant=dominant(data),
                average=float(scores.mean()) if len(rows) else None,
                median=float(scores.median()) if len(rows) else None,
                low=low, low_percentage=100 * low / len(rows) if len(rows) else 0,
                categories=rows.emotion_label.nunique())


def filtered_export(result, view: pd.DataFrame) -> bytes:
    return result.data.iloc[view.index].to_csv(index=False).encode('utf-8')
