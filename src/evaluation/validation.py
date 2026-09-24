from dataclasses import dataclass
import pandas as pd
from src.config import MODEL_EMOTIONS
from src.data.preparation import PreparationResult, prepare_feedback, _available_name
from src.data.validation import DatasetValidationError


@dataclass
class EvaluationInput:
    prepared: PreparationResult
    fields: dict[str, str]
    duplicate_count: int
    valid_count: int


def normalize_truth(value):
    if not isinstance(value, str):
        return None, 'missing_label' if pd.api.types.is_scalar(value) and pd.isna(value) else 'unsupported_label'
    normalized = value.strip().lower()
    if not normalized:
        return None, 'missing_label'
    if normalized not in MODEL_EMOTIONS:
        return None, 'unsupported_label'
    return normalized, 'valid'


def validate_evaluation(data: pd.DataFrame) -> EvaluationInput:
    missing = {'feedback', 'true_emotion'} - set(data.columns)
    if missing:
        raise DatasetValidationError('Evaluation CSV requires columns: feedback and true_emotion. Missing: ' + ', '.join(sorted(missing)))
    if data.empty:
        raise DatasetValidationError('Evaluation dataset is empty. Add labeled feedback rows.')
    prepared = prepare_feedback(data, 'feedback')
    frame = prepared.data
    truth = [normalize_truth(value) for value in data.true_emotion]
    statuses = [label_status if label_status != 'valid' else text_status
                for (_, label_status), text_status in zip(truth, frame[prepared.fields['feedback_status']])]
    fields = {}
    # Duplicate normalized text/label pairs count even if optional context differs.
    duplicate = pd.DataFrame({'text': frame[prepared.fields['feedback_clean']].to_numpy(),
                              'truth': [value for value, _ in truth]}).duplicated()
    for name, values in [('true_emotion_normalized', [v for v, _ in truth]),
                         ('evaluation_validation', statuses), ('evaluation_duplicate', duplicate.to_numpy())]:
        field = _available_name(frame, name)
        frame[field] = values
        fields[name] = field
    usable = [status == 'valid' for status in statuses]
    # Only the evaluation preparation copy is altered, never normal analysis.
    frame[prepared.fields['feedback_is_usable']] = usable
    return EvaluationInput(prepared, fields, int(duplicate.sum()), sum(usable))
