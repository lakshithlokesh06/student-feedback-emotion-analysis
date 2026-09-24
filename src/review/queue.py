from dataclasses import dataclass, field
import pandas as pd
from src.config import MODEL_EMOTIONS, HIGH_CONFIDENCE_THRESHOLD
from src.analytics.insights import ATTENTION_EMOTIONS
from src.review.identifiers import dataset_identity, row_identifiers


def create_queue(data, fields, source, feedback_column, contexts=None, revision=''):
    identity = dataset_identity(data, source, revision)
    ids = row_identifiers(data, source, identity, feedback_column)
    queue = pd.DataFrame({'review_id': ids, 'feedback': data[feedback_column].to_numpy(),
                          'source_dataset_type': source, 'dataset_identity': identity})
    prediction = fields.get('predicted_emotion', fields.get('emotion_label'))
    queue['predicted_emotion'] = data[prediction].to_numpy()
    queue['emotion_confidence'] = data[fields['emotion_confidence']].to_numpy()
    queue['low_confidence'] = data[fields['emotion_low_confidence']].to_numpy()
    for name in ('second_emotion', 'second_emotion_confidence', 'confidence_margin', 'ambiguous_prediction', 'was_truncated', 'token_length', 'is_correct'):
        if fields.get(name) in data:
            queue[name] = data[fields[name]].to_numpy()
    if fields.get('true_emotion_normalized') in data:
        queue['true_emotion'] = data[fields['true_emotion_normalized']].to_numpy()
        queue['evaluation_label_original'] = data['true_emotion'].to_numpy()
    for role, column in (contexts or {}).items():
        if column in data:
            queue[role] = data[column].to_numpy()
    valid = data[fields['emotion_status']].eq('analyzed').to_numpy()
    valid = valid & queue.predicted_emotion.isin(MODEL_EMOTIONS).to_numpy()
    return queue.loc[valid].set_index('review_id', drop=False), identity


@dataclass
class QueueFilters:
    statuses: list[str] = field(default_factory=list)
    predicted: list[str] = field(default_factory=list)
    reviewed: list[str] = field(default_factory=list)
    courses: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    semesters: list[str] = field(default_factory=list)
    confidence: tuple[float, float] = (0.0, 1.0)
    low_confidence: str = 'All'
    ambiguity: str = 'All'


def filter_queue(data, filters):
    mask = data.emotion_confidence.between(*filters.confidence)
    for column, choices in [('review_status', filters.statuses), ('predicted_emotion', filters.predicted), ('reviewed_emotion', filters.reviewed)]:
        if choices:
            mask &= data[column].isin(choices)
    for column, choices in [('course', filters.courses), ('subject', filters.subjects), ('semester', filters.semesters)]:
        if choices and column in data:
            mask &= data[column].astype('string').isin(choices)
    for column, choice in [('low_confidence', filters.low_confidence), ('ambiguous_prediction', filters.ambiguity)]:
        if column in data and choice != 'All':
            mask &= data[column].eq(choice == 'Yes').fillna(False)
    return data.loc[mask]


def prioritize(data, mode, emotion=None):
    if mode == 'Low-confidence predictions':
        return data.loc[data.low_confidence.eq(True)].sort_values('emotion_confidence', kind='stable')
    if mode == 'High-confidence predictions':
        return data.loc[data.emotion_confidence.ge(HIGH_CONFIDENCE_THRESHOLD)].sort_values('emotion_confidence', ascending=False, kind='stable')
    if mode == 'Ambiguous predictions':
        return data.loc[data.ambiguous_prediction.eq(True)].sort_values('confidence_margin', kind='stable') if 'ambiguous_prediction' in data else data.iloc[:0]
    if mode == 'Specific emotion':
        return data.loc[data.predicted_emotion.eq(emotion)]
    if mode == 'Feedback requiring attention':
        return data.loc[data.predicted_emotion.isin(ATTENTION_EMOTIONS)]
    if mode == 'Misclassified evaluation rows':
        return data.loc[data.is_correct.eq(False)] if 'is_correct' in data else data.iloc[:0]
    return data
