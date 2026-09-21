"""Resolve label IDs only through model metadata; never guess mappings."""
import re

from src.config import MODEL_EMOTIONS


class EmotionError(ValueError):
    """A safe, actionable model or inference failure."""


def normalize_label(label: str, id2label: dict) -> str:
    normalized = str(label).strip().lower()
    match = re.fullmatch(r'label_(\d+)', normalized)
    if match:
        key = int(match.group(1))
        normalized = str(id2label.get(key, id2label.get(str(key), ''))).strip().lower()
    if normalized not in MODEL_EMOTIONS:
        raise EmotionError('The model returned an undocumented emotion label. Check its label metadata before analyzing.')
    return normalized


def label_names(id2label: dict) -> list[str]:
    try:
        labels = [normalize_label(id2label.get(i, id2label.get(str(i), '')), id2label)
                  for i in range(len(id2label))]
    except (TypeError, AttributeError) as exc:
        raise EmotionError('The model has invalid label metadata.') from exc
    if set(labels) != set(MODEL_EMOTIONS) or len(labels) != len(MODEL_EMOTIONS):
        raise EmotionError('The model label configuration does not match the supported emotion categories.')
    return labels
