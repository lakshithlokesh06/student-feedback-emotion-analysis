import pandas as pd
from src.config import REVIEW_STATUSES, MODEL_EMOTIONS, EVALUATION_CONFIDENCE_EDGES


def progress(data):
    counts = data.review_status.value_counts()
    result = {status: int(counts.get(status, 0)) for status in REVIEW_STATUSES}
    result['total'] = len(data)
    result['reviewed'] = result['accepted'] + result['corrected'] + result['uncertain']
    result['remaining'] = result['unreviewed'] + result['skipped']
    result['completion'] = result['reviewed'] / len(data) if len(data) else 0.0
    return result


def agreement(data):
    decided = data.loc[data.review_status.isin(['accepted','corrected'])]
    accepted = int(decided.review_status.eq('accepted').sum())
    corrected = len(decided) - accepted
    summary = dict(accepted=accepted, corrected=corrected, agreement=accepted / len(decided) if len(decided) else None,
                   accepted_confidence=decided.loc[decided.review_status.eq('accepted'), 'emotion_confidence'].mean(),
                   corrected_confidence=decided.loc[decided.review_status.eq('corrected'), 'emotion_confidence'].mean())
    matrix = pd.crosstab(decided.predicted_emotion, decided.reviewed_emotion).reindex(index=MODEL_EMOTIONS, columns=MODEL_EMOTIONS, fill_value=0)
    changes = decided.loc[decided.review_status.eq('corrected')]
    transitions = changes.groupby(['predicted_emotion','reviewed_emotion']).size().reset_index(name='Count').sort_values('Count', ascending=False, kind='stable')
    return summary, matrix, transitions


def correction_bands(data):
    decided = data.loc[data.review_status.isin(['accepted','corrected'])].copy()
    edges = list(EVALUATION_CONFIDENCE_EDGES)
    labels = [f'[{lo:.2f}, {hi:.2f}{"]" if hi == 1 else ")"}' for lo, hi in zip(edges[:-1], edges[1:])]
    edges[-1] += 1e-12
    decided['Band'] = pd.cut(decided.emotion_confidence, edges, labels=labels, right=False, include_lowest=True)
    decided['Corrected'] = decided.review_status.eq('corrected')
    return decided.groupby('Band', observed=False).agg(Count=('Corrected','size'), Correction_rate=('Corrected','mean')).reset_index()


def annotation_comparison(data):
    if 'true_emotion' not in data:
        return pd.DataFrame(columns=['Comparison','Count'])
    rows = data.loc[data.review_status.isin(['accepted','corrected']) & data.true_emotion.notna()]
    labels = []
    for model, reference, human in zip(rows.predicted_emotion, rows.true_emotion, rows.reviewed_emotion):
        if model == reference == human:
            labels.append('All three agree')
        elif model == human:
            labels.append('Model and human agree; evaluation label differs')
        elif reference == human:
            labels.append('Evaluation label and human agree; model differs')
        elif model == reference:
            labels.append('Model and evaluation label agree; human differs')
        else:
            labels.append('All three differ')
    return pd.Series(labels, dtype='string').value_counts().rename_axis('Comparison').reset_index(name='Count')


def observations(data):
    stats = progress(data)
    summary, _, transitions = agreement(data)
    output = [f"{stats['uncertain']} reviewed items were marked uncertain; {stats['skipped']} were deferred."]
    if summary['agreement'] is not None:
        output.append(f"Model-human agreement is {summary['agreement']:.1%} among {summary['accepted'] + summary['corrected']} accepted/corrected decisions in this reviewed subset.")
    if not transitions.empty:
        maximum = transitions.Count.max()
        tied = transitions.loc[transitions.Count.eq(maximum)]
        output.append('Most frequent correction transition(s): ' + '; '.join(f'{row.predicted_emotion} → {row.reviewed_emotion} ({row.Count})' for row in tied.itertuples()) + '.')
    return output


def small_sample_warnings(data):
    decided = data.loc[data.review_status.isin(['accepted','corrected'])]
    messages = []
    if len(decided) < 10:
        messages.append('Fewer than 10 accepted/corrected decisions: agreement and correction patterns are preliminary.')
    if (decided.predicted_emotion.value_counts() < 3).any():
        messages.append('Some predicted emotions have fewer than three accepted/corrected reviews; avoid generalizing their correction rates.')
    return messages
