import pandas as pd
from src.analytics.trends import temporal_distribution
from src.analytics.metrics import analyzed


def temporal(data, frequency):
    composition, excluded = temporal_distribution(data, frequency)
    if composition.empty:
        return composition, pd.DataFrame(), excluded
    rows = analyzed(data)
    rows = rows.loc[rows.date.notna()].copy()
    rows['Period'] = rows.date.dt.tz_localize(None).dt.to_period({'Day':'D','Week':'W-SUN','Month':'M'}[frequency]).dt.start_time
    stats = rows.groupby('Period').agg(volume=('emotion_label','size'), average_confidence=('emotion_confidence','mean'), low_confidence_percentage=('emotion_low_confidence','mean')).reset_index()
    periods = pd.Index(composition.Period.unique(), name='Period')
    stats = stats.set_index('Period').reindex(periods).reset_index()
    stats['volume'] = stats.volume.fillna(0).astype(int)
    stats.low_confidence_percentage *= 100
    return composition, stats, excluded
