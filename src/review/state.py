"""Sparse latest decisions plus append-only save events for the current snapshot."""
from datetime import datetime, timezone
import pandas as pd
from src.review.queue import QueueFilters
from src.review.validation import validate_review, ReviewValidationError

ANNOTATION_FIELDS = ('review_status', 'reviewed_emotion', 'review_note', 'reviewed_at', 'reviewer_id')


def new_workspace(queue, identity, generation):
    return dict(queue=queue, identity=identity, generation=generation, records={}, events=[],
                filters=QueueFilters(), priority='All predictions', priority_emotion='joy',
                current_id=None, revision=0)


def save_review(workspace, review_id, status, reviewed_emotion=None, note='', reviewer='local_reviewer'):
    if review_id not in workspace['queue'].index:
        raise ReviewValidationError('This review item no longer belongs to the active prediction snapshot.')
    row = workspace['queue'].loc[review_id]
    annotation = validate_review(row.predicted_emotion, status, reviewed_emotion, note, reviewer)
    annotation['reviewed_at'] = datetime.now(timezone.utc).isoformat()
    workspace['records'][review_id] = annotation
    workspace['events'].append(dict(review_id=review_id, **annotation,
                                    original_predicted_emotion=row.predicted_emotion,
                                    original_prediction_confidence=float(row.emotion_confidence),
                                    source_dataset_type=row.source_dataset_type,
                                    dataset_identity=workspace['identity']))
    workspace['revision'] += 1


def annotated_queue(workspace):
    queue = workspace['queue'].copy(deep=False)
    annotations = pd.DataFrame.from_dict(workspace['records'], orient='index').reindex(index=queue.index, columns=ANNOTATION_FIELDS)
    annotations['review_status'] = annotations.review_status.fillna('unreviewed')
    return pd.concat([queue, annotations], axis=1)


def clear_progress(workspace):
    workspace['records'].clear()
    workspace['events'].clear()
    workspace['current_id'] = None
    workspace['revision'] += 1


def move_item(ids, current, offset):
    if not ids:
        return None
    index = ids.index(current) if current in ids else 0
    return ids[min(max(index + offset, 0), len(ids) - 1)]
