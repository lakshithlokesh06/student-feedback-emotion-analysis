from src.evaluation.runner import evaluated


def filter_errors(view, true_labels=(), predicted_labels=(), correctness='All', minimum_confidence=0.0):
    rows = evaluated(view)
    mask = rows.emotion_confidence.ge(minimum_confidence)
    if true_labels:
        mask &= rows.true_emotion_normalized.isin(true_labels)
    if predicted_labels:
        mask &= rows.predicted_emotion.isin(predicted_labels)
    if correctness != 'All':
        mask &= rows.is_correct.eq(correctness == 'Correct')
    return rows.loc[mask]


def high_confidence_errors(view, threshold):
    return filter_errors(view, correctness='Incorrect', minimum_confidence=threshold).sort_values('emotion_confidence', ascending=False)
