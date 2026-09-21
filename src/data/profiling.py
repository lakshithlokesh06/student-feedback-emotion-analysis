"""Small, bounded dataset summaries without an EDA dependency."""
from dataclasses import dataclass

import pandas as pd


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    columns: pd.DataFrame
    duplicate_rows: int
    context_unique: dict[str, int]


def profile_dataset(data: pd.DataFrame, contexts: dict[str, str | None] | None = None) -> DatasetProfile:
    columns = pd.DataFrame({
        "Column": list(data.columns),
        "Data type": [str(dtype) for dtype in data.dtypes],
        "Missing values": [int(data[column].isna().sum()) for column in data.columns],
        "Blank text": [int(data[column].map(lambda value: isinstance(value, str) and not value.strip()).sum()) for column in data.columns],
    })
    unique = {
        role: int(data[column].replace(r"^\s*$", pd.NA, regex=True).nunique(dropna=True))
        for role, column in (contexts or {}).items()
        if role in ("course", "subject", "semester") and column in data.columns
    }
    return DatasetProfile(len(data), len(data.columns), columns, int(data.duplicated().sum()), unique)
