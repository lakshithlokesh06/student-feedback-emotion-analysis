import re
import unicodedata
from src.config import MODEL_EMOTIONS, REVIEW_STATUSES, REVIEW_NOTE_MAX_LENGTH, REVIEWER_ID_MAX_LENGTH, DEFAULT_REVIEWER_ID


class ReviewValidationError(ValueError):
    """Safe annotation-validation message for display in the UI."""


def clean_note(note: str) -> str:
    if not isinstance(note, str):
        raise ReviewValidationError('Review note must be text.')
    note = unicodedata.normalize('NFC', note).replace('\r\n', '\n').replace('\r', '\n')
    note = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', note).strip()
    if len(note) > REVIEW_NOTE_MAX_LENGTH:
        raise ReviewValidationError(f'Review notes may contain at most {REVIEW_NOTE_MAX_LENGTH} characters.')
    return note


def validate_review(predicted: str, status: str, reviewed: str | None, note: str, reviewer: str) -> dict:
    if status not in REVIEW_STATUSES or status == 'unreviewed':
        raise ReviewValidationError('Choose accepted, corrected, uncertain, or skipped.')
    if predicted not in MODEL_EMOTIONS:
        raise ReviewValidationError('The original prediction has an unsupported emotion.')
    if status in ('accepted', 'corrected'):
        if reviewed not in MODEL_EMOTIONS:
            raise ReviewValidationError('Choose a supported reviewed emotion.')
        if (status == 'accepted') != (reviewed == predicted):
            raise ReviewValidationError('Accepted must match the prediction; corrected must use a different emotion.')
    else:
        reviewed = None
    if not isinstance(reviewer, str):
        raise ReviewValidationError('Reviewer identifier must be text.')
    reviewer = reviewer.strip() or DEFAULT_REVIEWER_ID
    if len(reviewer) > REVIEWER_ID_MAX_LENGTH or any(unicodedata.category(c).startswith('C') for c in reviewer):
        raise ReviewValidationError(f'Use a reviewer identifier of at most {REVIEWER_ID_MAX_LENGTH} characters without control characters.')
    return dict(review_status=status, reviewed_emotion=reviewed, review_note=clean_note(note), reviewer_id=reviewer)
