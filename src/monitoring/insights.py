def insights(details, js, band):
    rows = details.set_index('metric')
    volume = rows.loc['Feedback rows']
    confidence = rows.loc['Confidence mean']
    length = rows.loc['word_count median']
    messages = [f"Current feedback volume changed by {100*volume.change/volume.reference_value:+.1f}%.",
                f"Average confidence changed from {confidence.reference_value:.3f} to {confidence.current_value:.3f} ({100*confidence.change:+.1f} percentage points).",
                f"Emotion-distribution divergence is {js:.3f}: {band} under configured heuristic bands.",
                f"Median feedback length changed from {length.reference_value:g} to {length.current_value:g} words."]
    for row in details[details.metric.str.contains('prediction share')].itertuples():
        if abs(row.change) > 0.00001:
            messages.append(f'{row.metric} changed by {row.change:+.1f} percentage points.')
    return messages
