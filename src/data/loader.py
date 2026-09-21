"""Bounded, strict CSV intake. Preserve cell spelling rather than inferring types."""
import csv
import io
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.data.validation import DatasetValidationError, IntakeLimits, validate_headers


def load_csv(file: BinaryIO, filename: str, limits: IntakeLimits = IntakeLimits()) -> pd.DataFrame:
    if Path(filename).suffix.lower() != ".csv":
        raise DatasetValidationError("Unsupported file. Upload a file with a .csv extension.")
    try:
        file.seek(0)
        raw = file.read(limits.max_bytes + 1)
    except (OSError, ValueError) as exc:
        raise DatasetValidationError("Could not read the file. Please upload it again.") from exc
    if len(raw) > limits.max_bytes:
        raise DatasetValidationError(f"File exceeds the {limits.max_bytes / 1024 / 1024:g} MB limit.")
    if not raw:
        raise DatasetValidationError("The file is empty. Include a header and at least one feedback row.")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise DatasetValidationError("Could not decode this CSV. Save it with UTF-8 encoding and try again.") from exc
    if "\x00" in content:
        raise DatasetValidationError("CSV contains null bytes. Export a plain UTF-8 CSV.")
    reader = csv.reader(io.StringIO(content, newline=""), strict=True)
    rows = []
    try:
        headers = next(reader, [])
        validate_headers(headers, limits)
        for row in reader:
            # A blank physical record represents empty cells, not a discarded row.
            if not row:
                row = [""] * len(headers)
            if len(row) != len(headers):
                raise DatasetValidationError(f"CSV record ending at line {reader.line_num} has {len(row)} fields; expected {len(headers)}. Check delimiters and quoting.")
            rows.append(row)
            if len(rows) > limits.max_rows:
                raise DatasetValidationError(f"CSV exceeds the limit of {limits.max_rows:,} rows.")
    except csv.Error as exc:
        raise DatasetValidationError("Malformed CSV or an oversized cell. Check comma delimiters, matching quotes, and cell lengths.") from exc
    if not rows:
        raise DatasetValidationError("Dataset is empty. Add at least one data row below the header.")
    return pd.DataFrame(rows, columns=headers, dtype=object)
