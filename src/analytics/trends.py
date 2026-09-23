import pandas as pd
from src.config import MODEL_EMOTIONS
from src.analytics.metrics import analyzed


def default_frequency(data: pd.DataFrame) -> str:
    dates = pd.to_datetime(data.get('date', pd.Series(dtype='object')), errors='coerce', utc=True, format='ISO8601').dropna()
    days = (dates.max() - dates.min()).days if len(dates) else 0
    return 'Day' if days <= 31 else 'Week' if days <= 180 else 'Month'


def temporal_distribution(data: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, int]:
    if frequency not in ('Day', 'Week', 'Month'):
        raise ValueError('Choose Day, Week, or Month.')
    rows = analyzed(data)
    if 'date' not in rows:
        return pd.DataFrame(), len(rows)
    dates = pd.to_datetime(rows.date, errors='coerce', utc=True, format='ISO8601')
    excluded = int(dates.isna().sum())
    frame = rows.loc[dates.notna(), ['emotion_label']].copy()
    dates = dates.dropna().dt.tz_localize(None)
    frame['Period'] = dates.dt.to_period({'Day': 'D', 'Week': 'W-SUN', 'Month': 'M'}[frequency]).dt.start_time
    if frame.empty:
        return pd.DataFrame(), excluded
    counts = pd.crosstab(frame.Period, frame.emotion_label).reindex(columns=MODEL_EMOTIONS, fill_value=0)
    # Fill unobserved periods inside the observed range so gaps are not hidden.
    periods = pd.date_range(counts.index.min(), counts.index.max(), freq={'Day': 'D', 'Week': 'W-MON', 'Month': 'MS'}[frequency])
    if len(periods) > 600:
        raise ValueError('This range contains more than 600 periods. Choose a coarser time aggregation or narrow the dashboard filters.')
    counts = counts.reindex(periods, fill_value=0).rename_axis('Period')
    output = counts.reset_index().melt(id_vars='Period', var_name='Emotion', value_name='Count')
    output['Volume'] = output.groupby('Period').Count.transform('sum')
    output['Percentage'] = output.Count.div(output.Volume.replace(0, float('nan'))).mul(100)
    return output, excluded
