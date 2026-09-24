from src.evaluation.runner import evaluated


def quality_warnings(view, synthetic=False, duplicate_count=0):
    rows = evaluated(view)
    support = rows.true_emotion_normalized.value_counts()
    messages = []
    if synthetic:
        messages.append('This bundled dataset is synthetic and intended only to demonstrate the evaluation workflow. It is not a benchmark for real-world model accuracy.')
    if len(rows) < 30:
        messages.append('Fewer than 30 evaluated examples: metric estimates may be unstable.')
    if len(rows) < 100:
        messages.append('Fewer than 100 evaluated examples: calibration estimates are especially unstable; this does not validate calibration.')
    if (support < 3).any():
        messages.append('Some represented emotions have fewer than 3 examples. Treat their performance estimates cautiously.')
    if len(support) <= 2:
        messages.append('Only one or two emotions (or none) are represented; this cannot assess the full label set.')
    if len(support) and support.max() / support.sum() >= .70:
        messages.append('One emotion accounts for at least 70% of reference labels. Accuracy may hide class imbalance.')
    if duplicate_count:
        messages.append(f'{duplicate_count} duplicate normalized feedback/label pairs were retained; repeated examples can inflate apparent evidence.')
    return messages
