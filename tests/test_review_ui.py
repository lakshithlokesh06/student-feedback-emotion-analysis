from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.config import PROJECT_ROOT
from tests.test_emotion import FakeModel
from tests.test_evaluation import DetailedFake


def button(app,label):
    return next(item for item in app.button if item.label==label)


def radio(app,label):
    return next(item for item in app.radio if item.label==label)


def start_review():
    app=AppTest.from_file(str(PROJECT_ROOT/'app.py')).run()
    app.sidebar.radio[0].set_value('Analyze Feedback').run()
    with patch('src.ui.analysis.get_model',return_value=FakeModel()):
        button(app,'Analyze Feedback').click().run()
    app.sidebar.radio[0].set_value('Human Review').run()
    return app


def test_review_flow_persistence_and_invalidation():
    app=start_review()
    with patch('src.ui.analysis.get_model',side_effect=AssertionError('Review must not load a model')):
        button(app,'Save & Next').click().run()
        radio(app,'Review decision').set_value('Choose another emotion').run()
        next(item for item in app.selectbox if item.label=='Reviewed Emotion').select('fear').run()
        button(app,'Save & Next').click().run()
        radio(app,'Review decision').set_value('Mark uncertain').run()
        button(app,'Save & Next').click().run()
        assert not app.exception
        state=app.session_state['human_review']['workspaces']['Main Analysis Results']
        assert [row['review_status'] for row in state['records'].values()]==['accepted','corrected','uncertain']
        app.sidebar.radio[0].set_value('Emotion Dashboard').run()
        app.multiselect[1].select('Computer Science').run()
        app.sidebar.radio[0].set_value('Human Review').run()
        assert len(app.session_state['human_review']['workspaces']['Main Analysis Results']['records'])==3
        radio(app,'Review workspace').set_value('Agreement & Patterns').run()
        assert not app.exception
        radio(app,'Review workspace').set_value('Exports').run()
        assert len(app.get('download_button'))==3
        button(app,'Clear ALL saved review progress for this source').click().run()
        assert not app.session_state['human_review']['workspaces']['Main Analysis Results']['records']
    # Recreate a decision, then change the source data: it must not survive.
    radio(app,'Review workspace').set_value('Review Items').run()
    button(app,'First unreviewed item').click().run()
    button(app,'Save Review').click().run()
    app.sidebar.radio[0].set_value('Analyze Feedback').run()
    app.selectbox[0].select('rating').run()
    app.sidebar.radio[0].set_value('Human Review').run()
    assert 'Main Analysis Results' not in app.session_state['human_review']['workspaces']
    assert not app.exception


def test_review_sources_are_separate():
    app=start_review()
    button(app,'Save Review').click().run()
    app.sidebar.radio[0].set_value('Model Evaluation').run()
    with patch('src.ui.evaluation.get_model',return_value=DetailedFake()):
        button(app,'Run Evaluation').click().run()
    app.sidebar.radio[0].set_value('Human Review').run()
    radio(app,'Review source').set_value('Evaluation Results').run()
    button(app,'Save Review').click().run()
    spaces=app.session_state['human_review']['workspaces']
    assert len(spaces['Main Analysis Results']['records'])==1
    assert len(spaces['Evaluation Results']['records'])==1
    radio(app,'Review source').set_value('Main Analysis Results').run()
    assert len(app.session_state['human_review']['workspaces']['Main Analysis Results']['records'])==1
    assert not app.exception


def test_review_filters_and_rerun_invalidation():
    app = start_review()
    button(app, 'Save Review').click().run()
    next(item for item in app.multiselect if item.label == 'Review status').select('accepted').run()
    assert len(app.session_state['human_review']['workspaces']['Main Analysis Results']['records']) == 1
    button(app, 'Reset Filters').click().run()
    workspace = app.session_state['human_review']['workspaces']['Main Analysis Results']
    assert workspace['filters'].statuses == [] and len(workspace['records']) == 1
    app.sidebar.radio[0].set_value('Model Evaluation').run()
    with patch('src.ui.evaluation.get_model', return_value=DetailedFake()):
        button(app, 'Run Evaluation').click().run()
    app.sidebar.radio[0].set_value('Human Review').run()
    radio(app, 'Review source').set_value('Evaluation Results').run()
    button(app, 'Save Review').click().run()
    app.sidebar.radio[0].set_value('Model Evaluation').run()
    with patch('src.ui.evaluation.get_model', return_value=DetailedFake()):
        button(app, 'Run Evaluation').click().run()
    app.sidebar.radio[0].set_value('Human Review').run()
    assert not app.session_state['human_review']['workspaces']['Evaluation Results']['records']
    assert len(app.session_state['human_review']['workspaces']['Main Analysis Results']['records']) == 1
    app.sidebar.radio[0].set_value('Analyze Feedback').run()
    with patch('src.ui.analysis.get_model', return_value=FakeModel()):
        button(app, 'Analyze Feedback').click().run()
    app.sidebar.radio[0].set_value('Human Review').run()
    radio(app, 'Review source').set_value('Main Analysis Results').run()
    assert not app.session_state['human_review']['workspaces']['Main Analysis Results']['records']
    assert not app.exception
