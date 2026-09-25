import numpy as np
import pandas as pd
from src.config import MODEL_EMOTIONS, LOW_CONFIDENCE_THRESHOLD, AMBIGUITY_MARGIN_THRESHOLD
from src.data.validation import DatasetValidationError, validate_headers, IntakeLimits
from src.data.preparation import prepare_feedback


def validate_monitoring(data, threshold=LOW_CONFIDENCE_THRESHOLD):
    """Return an independent canonical view; retain unclassified rows for input drift."""
    validate_headers(list(data.columns), IntakeLimits())
    if data.empty:
        raise DatasetValidationError('Monitoring requires a non-empty dataset.')
    required = {'feedback', 'emotion_label', 'emotion_confidence'}
    if not required.issubset(data):
        raise DatasetValidationError('Analyzed predictions are required: feedback, emotion_label, emotion_confidence. Analyze raw feedback in Analyze Feedback, then load the main results here.')
    frame = data.copy(deep=True).reset_index(drop=True)
    labels = frame.emotion_label.astype('string').str.strip().str.lower().replace('', pd.NA)
    confidence = pd.to_numeric(frame.emotion_confidence, errors='coerce')
    present = labels.notna()
    raw_score = frame.emotion_confidence.astype('string').str.strip().fillna('').ne('')
    if (~labels[present].isin(MODEL_EMOTIONS)).any():
        raise DatasetValidationError('Unsupported emotion labels in monitoring data.')
    if (present & (~np.isfinite(confidence) | ~confidence.between(0, 1))).any() or (~present & raw_score).any():
        raise DatasetValidationError('Each prediction requires numeric confidence between 0 and 1; confidence without a label is invalid.')
    if not present.any():
        raise DatasetValidationError('No analyzed predictions are available.')
    prepared = prepare_feedback(frame, 'feedback', {c:c for c in ('rating','feedback_date') if c in frame})
    usable = prepared.data[prepared.fields['feedback_is_usable']]
    if (present & ~usable).any():
        raise DatasetValidationError('Predicted rows must contain usable feedback text (at least three characters).')
    frame['emotion_label'] = labels
    frame['emotion_confidence'] = confidence
    frame['emotion_status'] = np.where(present, 'analyzed', 'not_analyzed')
    frame['emotion_low_confidence'] = confidence.lt(threshold) & present
    clean = prepared.data[prepared.fields['feedback_clean']].where(usable)
    frame['character_count'] = clean.str.len()
    frame['word_count'] = clean.str.split().str.len()
    for role, field in [('date','feedback_date_parsed'), ('rating_numeric','rating_numeric')]:
        if field in prepared.fields:
            frame[role] = prepared.data[prepared.fields[field]]
    margin = None
    scores = [f'score_{emotion}' for emotion in MODEL_EMOTIONS]
    if all(c in frame for c in scores):
        values = frame.loc[present, scores].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1) & (values <= 1).all(axis=1) & np.isclose(values.sum(axis=1), 1, atol=1e-5)
        selected = np.array([MODEL_EMOTIONS.index(x) for x in labels[present]])
        valid &= np.isclose(values[np.arange(len(values)), selected], confidence[present], atol=1e-6)
        valid &= np.isclose(values.max(axis=1), confidence[present], atol=1e-6)
        if valid.all():
            ordered = np.sort(values, axis=1)
            margin = pd.Series(np.nan, index=frame.index)
            margin.loc[present] = ordered[:, -1] - ordered[:, -2]
    elif 'confidence_margin' in frame:
        candidate = pd.to_numeric(frame.confidence_margin, errors='coerce')
        if candidate[present].between(0, 1).all() and candidate[present].le(confidence[present]).all():
            margin = candidate
    # Never trust a precomputed boolean with an unknown ambiguity threshold.
    frame = frame.drop(columns=['ambiguous_prediction'], errors='ignore')
    if margin is not None:
        frame['ambiguous_prediction'] = margin.lt(AMBIGUITY_MARGIN_THRESHOLD) & present
    return frame


def from_main(state):
    result = state['analysis']
    frame = pd.DataFrame(index=result.data.index)
    frame['feedback'] = result.data[state['feedback_column']]
    for role, column in result.fields.items():
        frame[role] = result.data[column]
    for role, column in state.get('contexts', {}).items():
        if column in result.data:
            frame[role] = result.data[column]
    return frame
