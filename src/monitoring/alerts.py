import pandas as pd
from src.config import MONITORING_THRESHOLDS


def signals(details, js, rules=MONITORING_THRESHOLDS):
    flags = []
    def flag(metric, value, threshold, unit):
        flags.append(dict(level='review', signal='Monitoring signal detected', metric=metric, observed=value, threshold=threshold, unit=unit))
    if js >= rules['js_noticeable']:
        flag('Emotion-distribution divergence', js, rules['js_noticeable'], 'base-2 JSD')
    for row in details.itertuples():
        if row.metric == 'Confidence mean' and -row.change >= rules['confidence_drop']:
            flag(row.metric, row.change, rules['confidence_drop'], 'confidence drop')
        if row.metric == 'Low-confidence rate (%)' and row.change >= rules['low_increase_pp']:
            flag(row.metric, row.change, rules['low_increase_pp'], 'percentage-point increase')
        if row.metric.startswith('Missing ') and row.change >= rules['missing_increase_pp']:
            flag(row.metric, row.change, rules['missing_increase_pp'], 'percentage-point increase')
        if 'prediction share' in row.metric and abs(row.change) >= rules['emotion_change_pp']:
            flag(row.metric, row.change, rules['emotion_change_pp'], 'absolute percentage-point change')
        if row.metric == 'word_count median' and row.reference_value > 0 and abs(row.change / row.reference_value) >= rules['length_relative_change']:
            flag(row.metric, row.change / row.reference_value, rules['length_relative_change'], 'absolute relative change')
    return pd.DataFrame(flags, columns=['level','signal','metric','observed','threshold','unit'])
