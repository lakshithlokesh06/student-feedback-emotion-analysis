import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.config import MODEL_EMOTIONS
from src.evaluation.runner import evaluated


def performance(view):
    rows = evaluated(view)
    truth, prediction = rows.true_emotion_normalized, rows.predicted_emotion
    summary = {'accuracy': float(accuracy_score(truth, prediction)) if len(rows) else None}
    for average in ('macro', 'weighted'):
        values = precision_recall_fscore_support(truth, prediction, labels=list(MODEL_EMOTIONS), average=average, zero_division=0)[:3] if len(rows) else (None, None, None)
        summary.update({f'{average}_{metric}': value for metric, value in zip(('precision', 'recall', 'f1'), values)})
    p, r, f, support = precision_recall_fscore_support(truth, prediction, labels=list(MODEL_EMOTIONS), zero_division=0) if len(rows) else ([0.0] * len(MODEL_EMOTIONS),) * 3 + ([0] * len(MODEL_EMOTIONS),)
    classes = pd.DataFrame({'Emotion': MODEL_EMOTIONS, 'Precision': p, 'Recall': r, 'F1': f, 'Support': support})
    return summary, classes


def confusion(view, normalized=False):
    rows = evaluated(view)
    matrix = confusion_matrix(rows.true_emotion_normalized, rows.predicted_emotion, labels=list(MODEL_EMOTIONS)) if len(rows) else [[0] * len(MODEL_EMOTIONS) for _ in MODEL_EMOTIONS]
    frame = pd.DataFrame(matrix, index=MODEL_EMOTIONS, columns=MODEL_EMOTIONS)
    if normalized:
        frame = frame.div(frame.sum(axis=1).replace(0, float('nan')), axis=0).fillna(0) * 100
    return frame


def class_distribution(view):
    rows = evaluated(view)
    frames = []
    for column, name in [('true_emotion_normalized', 'True'), ('predicted_emotion', 'Predicted')]:
        counts = rows[column].value_counts().reindex(MODEL_EMOTIONS, fill_value=0)
        frame = counts.rename_axis('Emotion').reset_index(name='Count')
        frame['Percentage'] = frame.Count / len(rows) * 100 if len(rows) else 0
        frame['Labels'] = name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
