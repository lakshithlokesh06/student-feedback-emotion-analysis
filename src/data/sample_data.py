"""Load and validate bundled data without depending on the UI."""
from pathlib import Path

import pandas as pd

from src.config import REQUIRED_DATASET_COLUMNS, SAMPLE_DATASET_PATH


from src.data.validation import DatasetValidationError


def validate_sample_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the bundled schema and return the original frame without mutation."""
    missing = [column for column in REQUIRED_DATASET_COLUMNS if column not in data.columns]
    if missing:
        raise DatasetValidationError(f"Dataset is missing required columns: {', '.join(missing)}.")
    if data.empty:
        raise DatasetValidationError("Dataset is empty. Add at least one feedback row.")
    ids = data["feedback_id"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise DatasetValidationError("Feedback IDs must not be blank.")
    if ids.duplicated().any():
        raise DatasetValidationError("Dataset contains duplicated feedback IDs. Use a unique ID for each row.")
    usable = data["feedback"].map(lambda value: isinstance(value, str) and bool(value.strip()))
    if not usable.all():
        raise DatasetValidationError("Every feedback row must contain non-empty text.")
    return data


def load_sample_data(path: str | Path = SAMPLE_DATASET_PATH) -> pd.DataFrame:
    """Read a CSV, translating expected file/parser failures into helpful errors."""
    path = Path(path)
    if not path.is_file():
        raise DatasetValidationError(f"Sample dataset was not found at {path}.")
    try:
        data = pd.read_csv(path, dtype={"feedback_id": "string"})
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetValidationError(f"Could not read the sample dataset. Check that it is a readable UTF-8 CSV. Details: {exc}") from exc
    return validate_sample_data(data)
