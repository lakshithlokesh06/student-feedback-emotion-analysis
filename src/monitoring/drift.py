import pandas as pd
from src.config import MODEL_EMOTIONS, LOW_CONFIDENCE_THRESHOLD, MONITORING_THRESHOLDS
from src.analytics.metrics import analyzed, metrics
from src.monitoring.distributions import distribution, jensen_shannon, ks_distance


def severity(value, thresholds=MONITORING_THRESHOLDS):
    return 'substantial' if value >= thresholds['js_substantial'] else 'noticeable' if value >= thresholds['js_noticeable'] else 'minimal'


def missing(series):
    return series.isna() | series.astype('string').str.strip().eq('').fillna(False)


def describe(series):
    return {'mean': series.mean(), 'median': series.median(), 'q1': series.quantile(.25), 'q3': series.quantile(.75)}


def compare(reference, current, threshold=LOW_CONFIDENCE_THRESHOLD, rules=MONITORING_THRESHOLDS):
    if reference.empty or current.empty:
        raise ValueError('Both monitoring datasets must be non-empty.')
    a, b = (analyzed(frame).assign(emotion_low_confidence=lambda x: x.emotion_confidence.lt(threshold)) for frame in (reference, current))
    summaries = [metrics(x, threshold) for x in (reference, current)]
    emotions = distribution(a.emotion_label, b.emotion_label, MODEL_EMOTIONS, 'emotion')
    js = jensen_shannon(emotions.reference_count, emotions.current_count)
    ks = ks_distance(a.emotion_confidence, b.emotion_confidence)
    details = []
    def add(metric, av, bv, status='informational'):
        details.append(dict(metric=metric, reference_value=av, current_value=bv, change=bv-av, monitoring_status=status))
    # Distances compare each dataset to the reference; reference self-distance is zero.
    add('Emotion-distribution divergence', 0.0, js)
    add('Confidence KS distance', 0.0, ks)
    add('Feedback rows', len(reference), len(current))
    add('Analyzed rows', len(a), len(b))
    for stat in ('mean','median','q1','q3'):
        add('Confidence ' + stat, describe(a.emotion_confidence)[stat], describe(b.emotion_confidence)[stat])
    for label, key in [('Low-confidence count','low'), ('Low-confidence rate (%)','low_percentage')]:
        add(label, summaries[0][key], summaries[1][key])
    for col in ('character_count','word_count'):
        for stat in ('mean','median','q1','q3'):
            add(col + ' ' + stat, describe(reference[col])[stat], describe(current[col])[stat])
    for row in emotions.itertuples():
        add(row.emotion + ' prediction share (%)', row.reference_percentage, row.current_percentage)
    contexts = {}
    for col in ('course','subject','semester','rating'):
        if col in reference and col in current:
            if col == 'rating':
                av, bv = reference.rating_numeric, current.rating_numeric
                for stat in ('mean','median'):
                    add('Rating ' + stat, describe(av)[stat], describe(bv)[stat])
            else:
                av, bv = (x[col].astype('string').str.strip().replace('', pd.NA) for x in (reference,current))
            contexts[col] = distribution(av, bv)
    for col in ('feedback','course','subject','semester','rating','feedback_date'):
        if col in reference and col in current:
            add('Missing ' + col + ' (%)', missing(reference[col]).mean()*100, missing(current[col]).mean()*100)
    ambiguity = None
    if all('ambiguous_prediction' in x for x in (a,b)):
        add('Ambiguous count', int(a.ambiguous_prediction.sum()), int(b.ambiguous_prediction.sum()))
        add('Ambiguous rate (%)', a.ambiguous_prediction.mean()*100, b.ambiguous_prediction.mean()*100)
        ambiguity = by_emotion(a, b, 'ambiguous_prediction')
    from src.monitoring.alerts import signals
    table = pd.DataFrame(details)
    flags = signals(table, js, rules)
    table.loc[table.metric.isin(flags.metric), 'monitoring_status'] = 'review'
    from src.monitoring.insights import insights
    return dict(summaries=summaries, emotions=emotions, js=js, severity=severity(js,rules),
                ks=ks, details=table,
                contexts=contexts, ambiguity=ambiguity, low_by_emotion=by_emotion(a,b,'emotion_low_confidence'),
                signals=flags, insights=insights(table,js,severity(js,rules)))


def by_emotion(a, b, column):
    table = pd.DataFrame({'emotion': MODEL_EMOTIONS})
    for prefix, frame in [('reference',a),('current',b)]:
        group = frame.groupby('emotion_label')[column].agg(['size','sum','mean'])
        table[prefix+'_count'] = table.emotion.map(group['sum']).fillna(0).astype(int)
        table[prefix+'_support'] = table.emotion.map(group['size']).fillna(0).astype(int)
        table[prefix+'_percentage'] = table.emotion.map(group['mean']).mul(100).where(table[prefix+'_support'] >= MONITORING_THRESHOLDS['minimum_emotion_support'])
    return table
