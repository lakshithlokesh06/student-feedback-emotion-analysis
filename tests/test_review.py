from io import BytesIO
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from src.review.identifiers import dataset_identity, row_identifiers
from src.review.queue import create_queue, QueueFilters, filter_queue, prioritize
from src.review.state import new_workspace, save_review, annotated_queue, clear_progress, move_item
from src.review.validation import clean_note, validate_review, ReviewValidationError
from src.review.metrics import progress, agreement, correction_bands, annotation_comparison, observations, small_sample_warnings
from src.review.export import curated_export, reviewed_export, audit_export


@pytest.fixture
def workspace():
    frame = pd.DataFrame({'feedback_id':['a','b','c','d','e','f'], 'text':['A useful lesson','Worried by the exam','An ordinary lesson','Mixed feelings here','Another class',''],
        'label':['joy','fear','neutral','sadness','joy',None], 'score':[.9,.4,.85,.6,.7,None],
        'low':[False,True,False,False,False,None], 'status':['analyzed']*5+['not_analyzed'],
        'course':['A','B','B','A','C',None], 'subject':['X','Y','Y','X','Z',None], 'semester':['Fall']*6,
        'ambiguous':[False,True,False,False,False,None], 'margin':[.8,.05,.7,.2,.3,None],
        'reference':['joy','sadness','fear','neutral','anger',None], 'true_emotion':['joy','sadness','fear','neutral','anger',None],
        'correct':[True,False,False,False,False,None]})
    fields = dict(emotion_label='label',emotion_confidence='score',emotion_low_confidence='low',emotion_status='status',ambiguous_prediction='ambiguous',confidence_margin='margin',true_emotion_normalized='reference',is_correct='correct')
    queue, identity = create_queue(frame, fields, 'Evaluation Results','text', {'course':'course','subject':'subject','semester':'semester'}, 'pinned')
    state = new_workspace(queue, identity, 1)
    state['original_fixture'] = frame
    return state


def test_identifiers_stable_unique_and_snapshot_scoped():
    data = pd.DataFrame({'feedback_id':['x','x'], 'feedback':['Same','Same']}, index=[4,4])
    identity = dataset_identity(data, 'main')
    ids = row_identifiers(data, 'main', identity, 'feedback')
    assert len(set(ids)) == 2
    assert ids == row_identifiers(data.reset_index(drop=True), 'main', identity, 'feedback')
    assert identity == dataset_identity(data.reset_index(drop=True), 'main')
    assert dataset_identity(data, 'evaluation') != identity
    assert dataset_identity(data.assign(feedback=['Changed','Same']), 'main') != identity
    assert all(len(item) == 64 and 'Same' not in item for item in ids)


def test_queue_creation_and_original_integrity(workspace):
    original = workspace['original_fixture'].copy(deep=True)
    queue = workspace['queue']
    assert len(queue) == 5 and queue.index.is_unique
    assert set(queue.predicted_emotion) == {'joy','fear','neutral','sadness'}
    assert 'true_emotion' in queue and 'is_correct' in queue
    save_review(workspace, queue.index[0], 'corrected', 'neutral')
    assert_frame_equal(workspace['original_fixture'], original)
    assert queue.predicted_emotion.iloc[0] == 'joy' and queue.emotion_confidence.iloc[0] == .9


@pytest.mark.parametrize('status,label', [('accepted','joy'),('corrected','fear'),('uncertain',None),('skipped',None)])
def test_review_statuses(workspace,status,label):
    row_id = workspace['queue'].index[0]
    save_review(workspace,row_id,status,label,'  A note\x00  ','reviewer_1')
    row = annotated_queue(workspace).loc[row_id]
    assert row.review_status == status and row.review_note == 'A note'
    assert row.reviewed_emotion == label if label is not None else pd.isna(row.reviewed_emotion)
    assert row.reviewed_at.endswith('+00:00') and row.reviewer_id == 'reviewer_1'
    assert row.predicted_emotion == 'joy'


@pytest.mark.parametrize('status,label', [('accepted','fear'),('corrected','joy'),('corrected','frustration'),('unreviewed',None),('other',None)])
def test_invalid_decisions(status,label):
    with pytest.raises(ReviewValidationError):
        validate_review('joy',status,label,'','')


def test_note_validation():
    assert clean_note(' a\r\nb\x00 ') == 'a\nb'
    assert len(clean_note('x'*500)) == 500
    with pytest.raises(ReviewValidationError):
        clean_note('x'*501)
    with pytest.raises(ReviewValidationError):
        clean_note(3)
    with pytest.raises(ReviewValidationError):
        validate_review('joy','accepted','joy','','bad\nID')
    assert validate_review('joy','uncertain','fear','','')['reviewed_emotion'] is None


def decisions(workspace):
    ids = workspace['queue'].index
    save_review(workspace,ids[0],'accepted','joy')
    save_review(workspace,ids[1],'corrected','sadness')
    save_review(workspace,ids[2],'uncertain')
    save_review(workspace,ids[3],'skipped')
    return annotated_queue(workspace)


def test_progress_agreement_patterns(workspace):
    data = decisions(workspace)
    stats = progress(data)
    assert stats['reviewed'] == 3 and stats['remaining'] == 2 and stats['completion'] == .6
    summary, matrix, transitions = agreement(data)
    assert summary['agreement'] == .5
    assert summary['accepted_confidence'] == .9 and summary['corrected_confidence'] == .4
    assert matrix.loc['fear','sadness'] == 1 and matrix.to_numpy().sum() == 2
    assert transitions.iloc[0].to_dict() == {'predicted_emotion':'fear','reviewed_emotion':'sadness','Count':1}
    bands = correction_bands(data)
    assert bands.Count.sum() == 2 and bands.Correction_rate.iloc[0] == 1
    assert observations(data) == observations(data)
    assert len(small_sample_warnings(data)) == 2


@pytest.mark.parametrize('filters,count', [(QueueFilters(statuses=['unreviewed']),1),(QueueFilters(predicted=['joy']),2),
    (QueueFilters(reviewed=['sadness']),1),(QueueFilters(courses=['A']),2),(QueueFilters(subjects=['Y']),2),
    (QueueFilters(semesters=['Fall']),5),(QueueFilters(confidence=(.8,1)),2),(QueueFilters(low_confidence='Yes'),1),
    (QueueFilters(ambiguity='Yes'),1),(QueueFilters(statuses=['accepted'],predicted=['joy']),1)])
def test_filters(workspace,filters,count):
    data = decisions(workspace)
    assert len(filter_queue(data,filters)) == count
    assert len(workspace['records']) == 4


@pytest.mark.parametrize('mode,count', [('All predictions',5),('Low-confidence predictions',1),('High-confidence predictions',2),
    ('Specific emotion',2),('Feedback requiring attention',2),('Ambiguous predictions',1),('Misclassified evaluation rows',4)])
def test_priorities(workspace,mode,count):
    assert len(prioritize(annotated_queue(workspace),mode,'joy')) == count


def test_three_way_comparison(workspace):
    ids=workspace['queue'].index
    for item,status,label in [(0,'accepted','joy'),(1,'accepted','fear'),(2,'corrected','fear'),(3,'corrected','anger'),(4,'corrected','sadness')]:
        save_review(workspace,ids[item],status,label)
    comparisons=annotation_comparison(annotated_queue(workspace)).set_index('Comparison').Count
    assert comparisons['All three agree']==1
    assert comparisons['Model and human agree; evaluation label differs']==1
    assert comparisons['Evaluation label and human agree; model differs']==1
    assert comparisons['All three differ']==2
    data=annotated_queue(workspace).copy()
    data.loc[ids[3],'true_emotion']='sadness'
    assert 'Model and evaluation label agree; human differs' in annotation_comparison(data).Comparison.tolist()


def test_exports_and_audit_history(workspace):
    decisions(workspace)
    curated = pd.read_csv(BytesIO(curated_export(workspace)))
    assert curated.true_emotion.tolist()==['joy','sadness'] and len(curated)==2
    assert list(curated.columns)==['feedback','true_emotion','course','subject','semester']
    reviewed = pd.read_csv(BytesIO(reviewed_export(workspace)))
    assert len(reviewed)==4 and set(reviewed.review_status)=={'accepted','corrected','uncertain','skipped'}
    assert reviewed.original_predicted_emotion.tolist()==['joy','fear','neutral','sadness']
    save_review(workspace,workspace['queue'].index[0],'corrected','neutral',reviewer='second_local_id')
    audit = pd.read_csv(BytesIO(audit_export(workspace)))
    assert len(audit)==5 and len(workspace['records'])==4
    assert audit.reviewer_id.iloc[-1]=='second_local_id'
    assert audit.dataset_identity.eq(workspace['identity']).all()


def test_empty_one_and_clear(workspace):
    empty = new_workspace(workspace['queue'].iloc[:0],workspace['identity'],2)
    data=annotated_queue(empty)
    assert progress(data)['completion']==0 and agreement(data)[0]['agreement'] is None
    assert pd.read_csv(BytesIO(curated_export(empty))).empty
    assert move_item([],None,1) is None
    one=new_workspace(workspace['queue'].iloc[:1],workspace['identity'],2)
    item=one['queue'].index[0]
    save_review(one,item,'accepted','joy')
    assert agreement(annotated_queue(one))[0]['agreement']==1
    assert move_item([item],item,1)==item
    clear_progress(one)
    assert progress(annotated_queue(one))['unreviewed']==1 and not one['events']
    with pytest.raises(ReviewValidationError):
        save_review(one,'wrong','accepted','joy')
