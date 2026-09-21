"""Meaning-preserving feedback preparation; no emotion inference."""
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from src.config import MIN_FEEDBACK_LENGTH
from src.data.validation import DatasetValidationError, validate_selection


@dataclass
class PreparationResult:
    data: pd.DataFrame
    fields: dict[str, str]
    quality: dict[str, int]
    context_quality: dict[str, dict[str, int]]


def clean_feedback(value: object, minimum_length: int = MIN_FEEDBACK_LENGTH) -> tuple[str | None, str]:
    if not isinstance(value, str):
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return None, "missing"
        return None, "non_text"
    cleaned = re.sub(r"\s+", " ", value).strip()
    if not cleaned:
        return "", "empty"
    if len(cleaned) < minimum_length:
        return cleaned, "too_short"
    return cleaned, "valid"


def _available_name(data: pd.DataFrame, base: str) -> str:
    name, suffix = base, 2
    while name in data.columns:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def prepare_feedback(data: pd.DataFrame, feedback_column: str | None,
                     contexts: dict[str, str | None] | None = None,
                     minimum_length: int = MIN_FEEDBACK_LENGTH) -> PreparationResult:
    validate_selection(data, feedback_column)
    if minimum_length < 1:
        raise DatasetValidationError("Minimum feedback length must be at least one character.")
    contexts = contexts or {}
    for role, column in contexts.items():
        if column is not None:
            validate_selection(data, column, role)
    prepared = data.copy(deep=True)
    fields = {}

    def add(base: str, values) -> None:
        name = _available_name(prepared, base)
        prepared[name] = values
        fields[base] = name

    cleaned = [clean_feedback(value, minimum_length) for value in data[feedback_column]]
    statuses = [status for _, status in cleaned]
    add("feedback_clean", [text for text, _ in cleaned])
    add("feedback_status", statuses)
    add("feedback_is_usable", [status == "valid" for status in statuses])
    counts = pd.Series(statuses, dtype=object).value_counts()
    quality = {status: int(counts.get(status, 0)) for status in ("valid", "missing", "empty", "non_text", "too_short")}
    quality["total"] = len(data)
    context_quality = {}
    for role, base in (("feedback_date", "feedback_date_parsed"), ("rating", "rating_numeric")):
        column = contexts.get(role)
        if column is None:
            continue
        original = data[column]
        missing = original.isna() | original.map(lambda value: isinstance(value, str) and not value.strip())
        if role == "feedback_date":
            # Only ISO dates/timestamps: do not guess ambiguous locale dates or numeric epochs.
            text = original.astype("string").str.strip()
            iso = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$", na=False)
            converted = pd.to_datetime(text.where(iso), format="ISO8601", errors="coerce", utc=True)
        else:
            converted = pd.to_numeric(original.where(~missing), errors="coerce")
            converted = converted.where(np.isfinite(converted))
        add(base, converted)
        context_quality[role] = {
            "valid": int(converted.notna().sum()),
            "missing": int(missing.sum()),
            "invalid": int((~missing & converted.isna()).sum()),
        }
    return PreparationResult(prepared, fields, quality, context_quality)
