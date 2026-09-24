from dataclasses import dataclass
import pandas as pd
from src.config import MODEL_EMOTIONS, AMBIGUITY_MARGIN_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD
from src.data.preparation import _available_name
from src.emotion.classifier import classify_feedback, AnalysisResult
from src.evaluation.validation import EvaluationInput


@dataclass
class EvaluationResult:
    analysis: AnalysisResult
    fields: dict[str, str]
    ambiguity_threshold: float
    high_confidence_threshold: float
    duplicate_count: int

    def view(self) -> pd.DataFrame:
        """Canonical small view; positional indices select original export rows."""
        frame = pd.DataFrame(index=range(len(self.analysis.data)))
        for name, field in self.fields.items():
            frame[name] = self.analysis.data[field].to_numpy()
        return frame


def run_evaluation(source: EvaluationInput, model, progress=None) -> EvaluationResult:
    analysis = classify_feedback(source.prepared, model, progress=progress, include_details=True)
    data = analysis.data
    fields = dict(source.fields)
    fields.update(analysis.fields)
    fields['predicted_emotion'] = fields.pop('emotion_label')
    truth = data[source.fields['true_emotion_normalized']].tolist()
    predicted = data[fields['predicted_emotion']].tolist()
    statuses = data[fields['emotion_status']].tolist()
    score_rows = data[[fields[f'score_{label}'] for label in MODEL_EMOTIONS]].to_numpy()
    confidences = data[fields['emotion_confidence']].tolist()
    correct, second, second_scores, margins, ambiguous = [], [], [], [], []
    for i, status in enumerate(statuses):
        if status != 'analyzed':
            correct.append(None); second.append(None); second_scores.append(None); margins.append(None); ambiguous.append(None)
            continue
        ranked = sorted(zip(MODEL_EMOTIONS, score_rows[i]), key=lambda pair: -pair[1])
        runner_up = next(pair for pair in ranked if pair[0] != predicted[i])
        margin = float(confidences[i] - runner_up[1])
        correct.append(truth[i] == predicted[i]); second.append(runner_up[0]); second_scores.append(runner_up[1]); margins.append(margin)
        ambiguous.append(margin < AMBIGUITY_MARGIN_THRESHOLD)
    # Alias the prediction with the requested evaluation-specific field, preserving input collisions.
    for name, values in [('predicted_emotion', predicted), ('is_correct', correct), ('second_emotion', second),
                         ('second_emotion_confidence', second_scores), ('confidence_margin', margins), ('ambiguous_prediction', ambiguous)]:
        field = _available_name(data, name)
        data[field] = pd.array(values, dtype='boolean') if name in ('is_correct', 'ambiguous_prediction') else values
        fields[name] = field
    return EvaluationResult(analysis, fields, AMBIGUITY_MARGIN_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD, source.duplicate_count)


def evaluated(view):
    return view.loc[view.emotion_status.eq('analyzed')]


def export_results(result, positions=None) -> bytes:
    data = result.analysis.data if positions is None else result.analysis.data.iloc[positions]
    return data.to_csv(index=False).encode('utf-8')
