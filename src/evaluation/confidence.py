import numpy as np
import pandas as pd
from src.config import EVALUATION_CONFIDENCE_EDGES, CALIBRATION_BINS
from src.evaluation.runner import evaluated


def confidence_groups(view, edges=EVALUATION_CONFIDENCE_EDGES):
    rows = evaluated(view).copy()
    labels = [f'[{lo:.2f}, {hi:.2f}{"]" if hi == 1 else ")"}' for lo, hi in zip(edges[:-1], edges[1:])]
    boundaries = list(edges); boundaries[-1] = np.nextafter(1.0, 2.0)
    rows['Band'] = pd.cut(rows.emotion_confidence, boundaries, labels=labels, right=False, include_lowest=True)
    return rows.groupby('Band', observed=False).agg(Count=('is_correct', 'size'), Mean_confidence=('emotion_confidence', 'mean'), Accuracy=('is_correct', 'mean')).reset_index()


def correctness_confidence(view):
    summary = evaluated(view).groupby('is_correct').agg(Count=('emotion_confidence', 'size'), Mean_confidence=('emotion_confidence', 'mean')).reindex([True, False])
    summary['Count'] = summary.Count.fillna(0).astype(int)
    return summary.reset_index()


def calibration(view):
    return confidence_groups(view, tuple(np.linspace(0, 1, CALIBRATION_BINS + 1)))


def truncation_summary(view):
    rows = evaluated(view)
    return rows.groupby('was_truncated').agg(Count=('is_correct', 'size'), Accuracy=('is_correct', 'mean')).reset_index()
