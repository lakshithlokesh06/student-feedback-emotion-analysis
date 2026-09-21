"""Shared, UI-independent input validation."""
from dataclasses import dataclass

import pandas as pd

from src.config import MAX_COLUMNS, MAX_FILE_SIZE_BYTES, MAX_ROWS


class DatasetValidationError(ValueError):
    """An actionable input error that can be shown directly to the user."""


@dataclass(frozen=True)
class IntakeLimits:
    max_bytes: int = MAX_FILE_SIZE_BYTES
    max_rows: int = MAX_ROWS
    max_columns: int = MAX_COLUMNS


def validate_headers(headers: list[str], limits: IntakeLimits) -> None:
    if not headers:
        raise DatasetValidationError("CSV must contain at least one column and a header row.")
    if len(headers) > limits.max_columns:
        raise DatasetValidationError(f"CSV exceeds the limit of {limits.max_columns:,} columns.")
    normalized = [name.strip() for name in headers]
    if any(not name for name in normalized):
        raise DatasetValidationError("Column names cannot be blank. Name every column in the header row.")
    if len(set(normalized)) != len(normalized):
        raise DatasetValidationError("Duplicate column names are not supported, including names differing only by surrounding spaces. Rename them before uploading.")


def validate_selection(data: pd.DataFrame, column: str | None, label: str = "feedback") -> None:
    if column is None or column not in data.columns:
        raise DatasetValidationError(f"Select an existing {label} column to prepare feedback.")
