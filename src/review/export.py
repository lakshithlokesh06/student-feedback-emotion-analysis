import pandas as pd
from src.config import CONTEXT_COLUMNS
from src.review.state import annotated_queue


def reviewed_export(workspace):
    data = annotated_queue(workspace)
    data = data.loc[data.review_status.ne('unreviewed')].copy()
    data['original_predicted_emotion'] = data.predicted_emotion
    data['original_prediction_confidence'] = data.emotion_confidence
    return data.to_csv(index=False).encode('utf-8')


def curated_export(workspace):
    data = annotated_queue(workspace)
    decided = data.loc[data.review_status.isin(['accepted','corrected'])]
    columns = ['feedback', *(column for column in CONTEXT_COLUMNS if column in decided)]
    curated = decided[columns].copy()
    curated.insert(1, 'true_emotion', decided.reviewed_emotion)
    return curated.to_csv(index=False).encode('utf-8')


def audit_export(workspace):
    columns = ['review_id','source_dataset_type','dataset_identity','original_predicted_emotion',
               'original_prediction_confidence','reviewed_emotion','review_status','review_note','reviewed_at','reviewer_id']
    return pd.DataFrame(workspace['events'], columns=columns).to_csv(index=False).encode('utf-8')
