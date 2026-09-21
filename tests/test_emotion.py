from io import BytesIO
import os
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from src.config import MODEL_EMOTIONS, PROJECT_ROOT
from src.data.preparation import prepare_feedback
from src.emotion.classifier import classify_feedback, export_csv, summarize_results
from src.emotion.labels import EmotionError, label_names, normalize_label
from src.ui.intake_state import reset_dataset


class FakeModel:
    """Deterministic unit-test double, never used by application code."""
    metadata = {'labels': list(MODEL_EMOTIONS), 'name': 'unit-test-double', 'revision': 'test'}

    def __init__(self):
        self.calls = []

    def predict(self, texts):
        self.calls.append(texts)
        return [{'label': 'joy', 'score': 0.49 if text == 'Great!' else 0.8} for text in texts]


def prepared():
    return prepare_feedback(pd.DataFrame({'feedback': ['Great!', None, ' ', 'ok', 42, 'Interesting lesson', 'Thanks!']}), 'feedback')


def test_normalization_and_metadata():
    assert normalize_label(' JOY ', {}) == 'joy'
    assert normalize_label('LABEL_0', {0: 'sadness'}) == 'sadness'
    assert normalize_label('label_1', {'1': 'FEAR'}) == 'fear'
    assert label_names(dict(enumerate(MODEL_EMOTIONS))) == list(MODEL_EMOTIONS)
    for value in ['satisfaction', 'frustration', 'LABEL_99']:
        with pytest.raises(EmotionError):
            normalize_label(value, {})
    with pytest.raises(EmotionError):
        label_names({0: 'LABEL_0'})


def test_batches_fields_confidence_and_preservation():
    source = prepared()
    original = source.data.copy(deep=True)
    model = FakeModel()
    progress = []
    result = classify_feedback(source, model, batch_size=2, progress=progress.append)
    assert [len(batch) for batch in model.calls] == [2, 1]
    assert progress == [2 / 3, 1.0]
    assert_frame_equal(source.data, original)
    assert_frame_equal(result.data[original.columns], original)
    assert result.data.emotion_status.tolist() == ['analyzed', 'not_analyzed', 'not_analyzed', 'not_analyzed', 'not_analyzed', 'analyzed', 'analyzed']
    assert result.data.loc[0, 'emotion_confidence'] == 0.49
    assert bool(result.data.loc[0, 'emotion_low_confidence'])
    assert not bool(result.data.loc[5, 'emotion_low_confidence'])
    assert result.data.loc[1:4, 'emotion_label'].isna().all()
    assert result.data.loc[1:4, 'emotion_confidence'].isna().all()
    assert result.data.loc[1:4, 'emotion_low_confidence'].isna().all()
    assert result.metadata == model.metadata
    summary = summarize_results(result)
    assert (summary['analyzed'], summary['not_analyzed'], summary['dominant']) == (3, 4, 'joy')
    assert summary['distribution'].Percentage.sum() == 100


def test_threshold_boundary_and_field_collision():
    source = prepare_feedback(pd.DataFrame({'feedback': ['Great!'], 'emotion_label': ['original']}), 'feedback')
    result = classify_feedback(source, FakeModel(), threshold=0.49)
    assert not result.data.emotion_low_confidence.iloc[0]
    assert result.fields['emotion_label'] == 'emotion_label_2'
    assert result.data.emotion_label.iloc[0] == 'original'


def test_export_roundtrip():
    result = classify_feedback(prepared(), FakeModel())
    raw = export_csv(result)
    frame = pd.read_csv(BytesIO(raw))
    assert list(frame.columns) == list(result.data.columns)
    assert len(frame) == 7
    assert frame.emotion_confidence.iloc[0] == 0.49
    assert 'unit-test-double' not in raw.decode('utf-8')


def test_empty_usable_does_not_call_model():
    model = FakeModel()
    source = prepare_feedback(pd.DataFrame({'feedback': [None, '']}), 'feedback')
    result = classify_feedback(source, model)
    assert not model.calls
    assert result.data.emotion_status.eq('not_analyzed').all()
    assert summarize_results(result)['average_confidence'] is None


@pytest.mark.parametrize('output', [[{'label': 'joy', 'score': float('nan')}], [{'label': 'joy', 'score': 1.1}], [], [{'label': 'unknown', 'score': .5}]])
def test_bad_model_outputs_fail_atomically(output):
    model = FakeModel()
    model.predict = lambda texts: output
    with pytest.raises(EmotionError):
        classify_feedback(prepare_feedback(pd.DataFrame({'feedback': ['Hello']}), 'feedback'), model)


def test_model_failure_friendly():
    model = FakeModel()
    def fail(texts):
        raise RuntimeError('raw private implementation details')
    model.predict = fail
    with pytest.raises(EmotionError, match='No partial results'):
        classify_feedback(prepared(), model)


def test_source_invalidation():
    state = {'analysis': 'old', 'analysis_status': 'complete'}
    reset_dataset(state, 'Upload CSV', 'new', pd.DataFrame({'feedback': ['Hello']}))
    assert state['analysis'] is None
    assert state['analysis_status'] == 'not_analyzed'


def test_ui_results_navigation_and_invalidation():
    model = FakeModel()
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.radio[0].set_value('Analyze Feedback').run()
    with patch('src.ui.analysis.get_model', return_value=model) as loader:
        app.button[0].click().run()
        assert not app.exception
        assert app.session_state['intake']['analysis_status'] == 'complete'
        app.sidebar.radio[0].set_value('Emotion Dashboard').run()
        assert not app.exception
        assert len(app.get('plotly_chart')) == 2
        app.sidebar.radio[0].set_value('Overview').run()
        app.sidebar.radio[0].set_value('Analyze Feedback').run()
        assert app.session_state['intake']['analysis'] is not None
        assert loader.call_count == 1
        app.selectbox[0].select('rating').run()
        assert app.session_state['intake']['analysis'] is None
        assert not app.exception


@pytest.mark.skipif(os.environ.get('RUN_MODEL_INTEGRATION') != '1', reason='Opt-in real model download and CPU inference')
def test_real_model_smoke():
    from src.emotion.model_loader import load_model
    data = pd.DataFrame({'feedback': ['I loved the practical lesson!', 'I am worried about failing the exam.', 'Thanks! ' * 1000]})
    model = load_model()
    result = classify_feedback(prepare_feedback(data, 'feedback'), model, batch_size=2)
    assert len(result.data) == 3
    assert result.data.emotion_label.isin(MODEL_EMOTIONS).all()
    assert result.data.emotion_confidence.between(0, 1).all()
    print(result.data[['emotion_label', 'emotion_confidence']].to_string(index=False))


def test_model_loader_missing_dependencies():
    from src.emotion.model_loader import load_model
    with patch.dict('sys.modules', {'transformers': None}):
        with pytest.raises(EmotionError, match='dependencies are unavailable'):
            load_model()


def test_model_loader_metadata_and_cpu_configuration():
    from unittest.mock import MagicMock
    from src.emotion.model_loader import load_model
    tokenizer = MagicMock(model_max_length=512)
    model = MagicMock()
    model.config.id2label = dict(enumerate(MODEL_EMOTIONS))
    model.config._commit_hash = 'verified-revision'
    model.to.return_value = model
    mocked_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer)),
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=MagicMock(return_value=model)))
    with patch.dict('sys.modules', {'transformers': mocked_transformers}):
        loaded = load_model()
        assert loaded.metadata['labels'] == list(MODEL_EMOTIONS)
        assert loaded.metadata['revision'] == 'verified-revision'
        assert loaded.metadata['device'] == 'cpu'
        assert loaded.metadata['max_tokens'] == 512
        model.to.assert_called_once_with('cpu')
        model.eval.assert_called_once()
        mocked_transformers.AutoTokenizer.from_pretrained.side_effect = OSError('private detail')
        with pytest.raises(EmotionError, match='Could not load'):
            load_model()


def test_resource_cache_loads_once():
    from src.ui.model_cache import get_model
    get_model.clear()
    try:
        with patch('src.ui.model_cache.load_model', return_value=FakeModel()) as loader:
            assert get_model() is get_model()
            assert loader.call_count == 1
    finally:
        get_model.clear()
