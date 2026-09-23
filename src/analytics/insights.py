from src.analytics.metrics import analyzed, metrics
from src.analytics.distributions import emotion_distribution, confidence_by_emotion

ATTENTION_EMOTIONS = ('anger', 'fear', 'sadness', 'disgust')


def attention_feedback(data):
    return analyzed(data).loc[lambda rows: rows.emotion_label.isin(ATTENTION_EMOTIONS)].sort_values('emotion_confidence', ascending=False, kind='stable')


def descriptive_insights(data, threshold):
    stats = metrics(data, threshold)
    if not stats['analyzed']:
        return ['No analyzed feedback matches the current filters.']
    distribution = emotion_distribution(data)
    maximum = int(distribution.Count.max())
    share = maximum / stats['analyzed'] * 100
    output = [f"Most common predicted emotion(s): {stats['dominant']}; each appears in {share:.1f}% of {stats['analyzed']} analyzed responses.",
              f"Low-confidence predictions represent {stats['low_percentage']:.1f}% ({stats['low']} responses) of analyzed feedback."]
    confidence = confidence_by_emotion(data)
    highest = confidence.loc[confidence['mean'].eq(confidence['mean'].max())]
    output.append(f"Highest average model confidence: {', '.join(highest.emotion_label)} ({highest['mean'].iloc[0]:.1%}). Confidence is not validated accuracy.")
    output.append('These observations describe the current filtered view; they do not establish causes or student wellbeing.')
    return output
