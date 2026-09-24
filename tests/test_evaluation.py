from io import BytesIO
from unittest.mock import patch
import os
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from src.config import MODEL_EMOTIONS, LABELED_SAMPLE_PATH, PROJECT_ROOT
from src.data.loader import load_csv
from src.data.validation import DatasetValidationError
from src.evaluation.validation import normalize_truth, validate_evaluation
from src.evaluation.runner import run_evaluation, export_results
from src.evaluation.metrics import performance, confusion, class_distribution
from src.evaluation.confidence import confidence_groups, correctness_confidence, calibration, truncation_summary
from src.evaluation.errors import filter_errors, high_confidence_errors
from src.evaluation.quality import quality_warnings
from src.evaluation.state import replace_dataset


class DetailedFake:
    """Offline test double; application always uses the real cached model."""
    metadata = {'labels': list(MODEL_EMOTIONS), 'max_tokens': 512, 'revision': 'test-only'}
    def __init__(self):
        self.calls = []
    def predict_detailed(self, texts):
        self.calls.append(texts)
        output = []
        for text in texts:
            scores = dict.fromkeys(MODEL_EMOTIONS, 0.0)
            scores.update(joy=.51, fear=.49) if text == 'Ambiguous response' else scores.update(joy=.9, fear=.1)
            output.append({'label': 'joy', 'score': scores['joy'], 'scores': scores, 'token_length': 600 if len(text) > 100 else 12, 'was_truncated': len(text) > 100})
        return output


def source():
    return validate_evaluation(pd.DataFrame({'feedback': ['Good tutorial','Ambiguous response','Another response',None,'ok','Unknown label'],
                                           'true_emotion': [' JOY ', 'fear', 'sadness', 'joy', 'joy', 'positive']}))


@pytest.mark.parametrize('value,expected', [(' JOY ', ('joy','valid')), ('frustration',(None,'unsupported_label')), ('satisfaction',(None,'unsupported_label')), ('negative',(None,'unsupported_label')), (None,(None,'missing_label')), (' ',(None,'missing_label')), (42,(None,'unsupported_label'))])
def test_truth_normalization(value, expected):
    assert normalize_truth(value) == expected


def test_validation_required_empty_and_sample():
    with pytest.raises(DatasetValidationError, match='requires'):
        validate_evaluation(pd.DataFrame({'feedback':['Hello']}))
    with pytest.raises(DatasetValidationError, match='empty'):
        validate_evaluation(pd.DataFrame(columns=['feedback','true_emotion']))
    with LABELED_SAMPLE_PATH.open('rb') as file:
        sample = load_csv(file, LABELED_SAMPLE_PATH.name)
    assert 40 <= len(sample) <= 60
    assert set(sample.true_emotion) == set(MODEL_EMOTIONS)
    assert validate_evaluation(sample).valid_count == len(sample)
    assert not sample.duplicated().any()


def test_validation_preserves_and_rejects():
    data = pd.DataFrame({'feedback': ['Hello', '  Hello ', 42, ' ', 'Yes'], 'true_emotion': ['JOY',' joy ', 'fear','joy',None]})
    before = data.copy(deep=True)
    validated = validate_evaluation(data)
    assert validated.valid_count == 2
    assert validated.duplicate_count == 1
    assert_frame_equal(data, before)
    assert_frame_equal(validated.prepared.data[data.columns], before)


def test_evaluation_fields_scores_correctness_and_export():
    model = DetailedFake()
    validated = source()
    before = validated.prepared.data.copy(deep=True)
    result = run_evaluation(validated, model)
    view = result.view()
    assert len(model.calls[0]) == 3
    assert view.loc[:2,'is_correct'].tolist() == [True,False,False]
    assert view.loc[3:,'emotion_status'].eq('not_analyzed').all()
    assert view.loc[3:,'is_correct'].isna().all()
    assert view.loc[1,'confidence_margin'] == pytest.approx(.02)
    assert bool(view.loc[1,'ambiguous_prediction'])
    assert view.loc[0,'second_emotion'] == 'fear'
    assert view.loc[0,'second_emotion_confidence'] == .1
    assert_frame_equal(validated.prepared.data, before)
    frame = pd.read_csv(BytesIO(export_results(result)))
    assert len(frame) == 6 and 'is_correct' in frame and 'predicted_emotion' in frame
    assert frame.true_emotion.iloc[0] == ' JOY '
    errors = high_confidence_errors(view, .85)
    assert len(errors) == 1
    exported = pd.read_csv(BytesIO(export_results(result, view.loc[view.is_correct.eq(False)].index)))
    assert len(exported) == 2 and not exported.is_correct.any()


def test_exact_metrics_confusion_and_distributions():
    view = run_evaluation(source(), DetailedFake()).view()
    stats, classes = performance(view)
    assert stats['accuracy'] == pytest.approx(1/3)
    assert stats['macro_precision'] == pytest.approx(1/21)
    assert stats['macro_recall'] == pytest.approx(1/7)
    assert stats['macro_f1'] == pytest.approx(.5/7)
    assert stats['weighted_precision'] == pytest.approx(1/9)
    assert stats['weighted_recall'] == pytest.approx(1/3)
    assert stats['weighted_f1'] == pytest.approx(1/6)
    assert classes.Support.sum() == 3
    assert classes.set_index('Emotion').loc['anger','Support'] == 0
    matrix = confusion(view)
    assert matrix.loc['fear','joy'] == 1 and matrix.to_numpy().sum() == 3
    assert confusion(view, True).loc['fear','joy'] == 100
    dist = class_distribution(view)
    assert dist.groupby('Labels').Count.sum().eq(3).all()
    assert dist.groupby('Labels').Percentage.sum().round(8).eq(100).all()
    assert len(pd.read_csv(BytesIO(classes.to_csv(index=False).encode('utf-8')))) == 7


def test_confidence_boundaries_calibration():
    scores = [0.0,.4999,.50,.6999,.70,.8499,.85,1.0]
    view = pd.DataFrame({'emotion_status':['analyzed']*8, 'emotion_confidence':scores,'is_correct':[True,False]*4})
    bands = confidence_groups(view)
    assert bands.Count.tolist() == [2,2,2,2]
    assert bands.Accuracy.eq(.5).all()
    assert calibration(view).Count.sum() == 8
    stats = correctness_confidence(view).set_index('is_correct')
    assert stats.loc[True,'Mean_confidence'] == pytest.approx(sum(scores[::2])/4)
    assert stats.loc[False,'Mean_confidence'] == pytest.approx(sum(scores[1::2])/4)


def test_filters_warnings_truncation():
    result = run_evaluation(source(), DetailedFake())
    view = result.view()
    assert len(filter_errors(view, true_labels=['fear'], predicted_labels=['joy'], correctness='Incorrect', minimum_confidence=.5)) == 1
    assert truncation_summary(view).Count.sum() == 3
    assert len(quality_warnings(view, synthetic=True, duplicate_count=1)) >= 4
    assert any('synthetic' in msg for msg in quality_warnings(view, True))
    assert any('70%' in msg for msg in quality_warnings(view.iloc[:1]))


def test_empty_evaluated_and_one_row():
    invalid = validate_evaluation(pd.DataFrame({'feedback':['x'], 'true_emotion':['joy']}))
    result = run_evaluation(invalid, None)
    stats, classes = performance(result.view())
    assert stats['accuracy'] is None and classes.Support.sum() == 0
    assert confusion(result.view()).to_numpy().sum() == 0
    assert confidence_groups(result.view()).Count.sum() == 0
    assert class_distribution(result.view()).Count.sum() == 0
    one = run_evaluation(validate_evaluation(pd.DataFrame({'feedback':['Hello'], 'true_emotion':['joy']})), DetailedFake()).view()
    assert performance(one)[0]['accuracy'] == 1


def test_state_independence():
    main = {'analysis': 'unchanged'}
    state = {'results':'old', 'filters':{'true':['joy']}}
    replace_dataset(state, 'Upload labeled CSV', 'new', source())
    assert state['results'] is None and state['filters']['true'] == []
    assert main == {'analysis':'unchanged'}


def test_evaluation_ui_sections_navigation_and_isolation():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.radio[0].set_value('Model Evaluation').run()
    with patch('src.ui.evaluation.get_model', return_value=DetailedFake()) as loader:
        app.button[0].click().run()
        assert not app.exception
        assert app.session_state['evaluation']['status'] == 'complete'
        for section in ['Performance','Confidence & Calibration','Error Analysis','Score Transparency']:
            app.main.radio[1].set_value(section).run()
            assert not app.exception, section
        app.main.radio[1].set_value('Error Analysis').run()
        app.multiselect[0].select('fear').run()
        app.sidebar.radio[0].set_value('Analyze Feedback').run()
        app.main.radio[0].set_value('Upload CSV').run()
        app.sidebar.radio[0].set_value('Model Evaluation').run()
        assert app.session_state['evaluation']['results'] is not None
        assert app.session_state['evaluation']['filters']['true'] == ['fear']
        assert loader.call_count == 1
        app.main.radio[0].set_value('Upload labeled CSV').run()
        assert app.session_state['evaluation']['results'] is None
        assert not app.exception


@pytest.mark.skipif(os.environ.get('RUN_MODEL_INTEGRATION') != '1', reason='Opt-in real evaluation smoke test')
def test_real_evaluation():
    from src.emotion.model_loader import load_model
    model = load_model()
    with LABELED_SAMPLE_PATH.open('rb') as file:
        dataset = validate_evaluation(load_csv(file, LABELED_SAMPLE_PATH.name))
    result = run_evaluation(dataset, model)
    assert result.view().is_correct.notna().sum() == 49
    scores = result.view()[[f'score_{label}' for label in MODEL_EMOTIONS]]
    assert scores.sum(axis=1).between(.99999,1.00001).all()
    text = ['I enjoyed the seminar.']
    normal, detailed = model.predict(text)[0], model.predict_detailed(text)[0]
    assert normal['label'] == detailed['label']
    assert normal['score'] == pytest.approx(detailed['score'], abs=1e-8)
    long_result = model.predict_detailed(['Detailed practical feedback ' * 1000])[0]
    assert long_result['was_truncated'] and long_result['token_length'] > 512
    print(performance(result.view())[0])


def test_output_collisions_and_truncation():
    data = pd.DataFrame({'feedback':['Extended feedback ' * 40], 'true_emotion':['joy'],
                         'predicted_emotion':['original value'], 'is_correct':['original flag']})
    result = run_evaluation(validate_evaluation(data), DetailedFake())
    assert result.analysis.data.predicted_emotion.iloc[0] == 'original value'
    assert result.analysis.data.is_correct.iloc[0] == 'original flag'
    assert result.fields['predicted_emotion'] != 'predicted_emotion'
    assert result.view().was_truncated.iloc[0]
    assert truncation_summary(result.view()).set_index('was_truncated').loc[True, 'Count'] == 1


def test_evaluation_changes_preserve_main_predictions():
    from tests.test_emotion import FakeModel
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.radio[0].set_value('Analyze Feedback').run()
    with patch('src.ui.analysis.get_model', return_value=FakeModel()):
        app.button[0].click().run()
    main = app.session_state['intake']['analysis'].data.copy(deep=True)
    app.sidebar.radio[0].set_value('Model Evaluation').run()
    with patch('src.ui.evaluation.get_model', return_value=DetailedFake()):
        app.button[0].click().run()
    app.main.radio[0].set_value('Upload labeled CSV').run()
    assert_frame_equal(app.session_state['intake']['analysis'].data, main)
    assert app.session_state['evaluation']['results'] is None
    assert not app.exception
