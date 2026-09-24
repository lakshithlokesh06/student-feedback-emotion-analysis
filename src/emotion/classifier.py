"""Batched, atomic classification and export of prepared feedback."""
from dataclasses import dataclass
import math
from typing import Callable

import pandas as pd

from src.config import INFERENCE_BATCH_SIZE, LOW_CONFIDENCE_THRESHOLD, MODEL_EMOTIONS
from src.data.preparation import PreparationResult, _available_name
from src.emotion.labels import EmotionError, normalize_label


@dataclass
class AnalysisResult:
    data: pd.DataFrame
    fields: dict[str, str]
    metadata: dict
    threshold: float


def classify_feedback(prepared: PreparationResult, model, batch_size: int = INFERENCE_BATCH_SIZE,
                      threshold: float = LOW_CONFIDENCE_THRESHOLD,
                      progress: Callable[[float], None] | None = None,
                      include_details: bool = False) -> AnalysisResult:
    if batch_size < 1 or not 0 <= threshold <= 1:
        raise EmotionError('Batch size must be positive and confidence threshold must be between zero and one.')
    data = prepared.data.copy(deep=True)
    usable = data[prepared.fields['feedback_is_usable']].eq(True).fillna(False).to_numpy()
    positions = [i for i, valid in enumerate(usable) if valid]
    values = {'emotion_label': [None] * len(data), 'emotion_confidence': [float('nan')] * len(data),
              'emotion_status': ['not_analyzed'] * len(data), 'emotion_low_confidence': [None] * len(data)}
    if include_details:
        for field in [*(f'score_{label}' for label in MODEL_EMOTIONS), 'token_length', 'was_truncated']:
            values[field] = [None] * len(data)
    if positions and model is None:
        raise EmotionError('Load the emotion model before analyzing usable feedback.')
    try:
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start:start + batch_size]
            texts = data.iloc[batch_positions][prepared.fields['feedback_clean']].tolist()
            predictions = model.predict_detailed(texts) if include_details else model.predict(texts)
            if len(predictions) != len(texts):
                raise EmotionError('The model returned an incomplete batch. No results were saved; retry analysis.')
            id2label = dict(enumerate(model.metadata['labels']))
            for position, prediction in zip(batch_positions, predictions):
                label = normalize_label(prediction['label'], id2label)
                score = float(prediction['score'])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise EmotionError('The model returned invalid confidence scores. No results were saved.')
                values['emotion_label'][position] = label
                values['emotion_confidence'][position] = score
                values['emotion_status'][position] = 'analyzed'
                values['emotion_low_confidence'][position] = score < threshold
                if include_details:
                    scores = prediction['scores']
                    if set(scores) != set(MODEL_EMOTIONS) or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in scores.values()):
                        raise EmotionError('The model returned invalid class probabilities. No results were saved.')
                    if not math.isclose(sum(scores.values()), 1.0, abs_tol=1e-5) or not math.isclose(scores[label], score, abs_tol=1e-6) or score < max(scores.values()) - 1e-6:
                        raise EmotionError('Class probabilities do not match the prediction. No results were saved.')
                    for emotion in MODEL_EMOTIONS:
                        values[f'score_{emotion}'][position] = float(scores[emotion])
                    values['token_length'][position] = int(prediction['token_length'])
                    values['was_truncated'][position] = bool(prediction['was_truncated'])
            if progress:
                progress(min((start + len(texts)) / len(positions), 1.0))
    except EmotionError:
        raise
    except Exception as exc:
        raise EmotionError('Emotion inference failed. Try a smaller dataset or batch size and retry. No partial results were saved.') from exc
    fields = {}
    for base, value in values.items():
        name = _available_name(data, base)
        data[name] = pd.array(value, dtype='boolean') if base == 'emotion_low_confidence' else value
        fields[base] = name
    return AnalysisResult(data, fields, dict(model.metadata) if model else {}, threshold)


def summarize_results(result: AnalysisResult) -> dict:
    frame, fields = result.data, result.fields
    analyzed = frame.loc[frame[fields['emotion_status']].eq('analyzed')]
    counts = analyzed[fields['emotion_label']].value_counts()
    distribution = counts.rename_axis('Emotion').reset_index(name='Count')
    distribution['Percentage'] = distribution['Count'] / len(analyzed) * 100 if len(analyzed) else pd.Series(dtype=float)
    dominant = ', '.join(sorted(counts[counts == counts.max()].index)) if len(counts) else 'Not yet analyzed'
    return {'total': len(frame), 'analyzed': len(analyzed), 'not_analyzed': len(frame) - len(analyzed),
            'dominant': dominant, 'categories': len(counts),
            'average_confidence': float(analyzed[fields['emotion_confidence']].mean()) if len(analyzed) else None,
            'distribution': distribution}


def export_csv(result: AnalysisResult) -> bytes:
    """Only flat dataframe fields; model objects and metadata are never exported."""
    return result.data.to_csv(index=False).encode('utf-8')
