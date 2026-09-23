from dataclasses import dataclass, field
import pandas as pd


@dataclass
class DashboardFilters:
    emotions: list[str] = field(default_factory=list)
    courses: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    semesters: list[str] = field(default_factory=list)
    ratings: list[float] = field(default_factory=list)
    confidence: str = 'All'


def apply_filters(data: pd.DataFrame, filters: DashboardFilters, threshold: float) -> pd.DataFrame:
    mask = pd.Series(True, index=data.index)
    for column, values in (('emotion_label', filters.emotions), ('course', filters.courses),
                           ('subject', filters.subjects), ('semester', filters.semesters), ('rating', filters.ratings)):
        if values and column in data:
            mask &= data[column].isin(values)
    if filters.confidence != 'All':
        mask &= data.emotion_status.eq('analyzed')
        mask &= data.emotion_confidence.lt(threshold) if filters.confidence == 'Low confidence' else data.emotion_confidence.ge(threshold)
    return data.loc[mask]
