from io import BytesIO
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT


def app():
    result = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    result.sidebar.radio[0].set_value('Analyze Feedback').run()
    return result


def test_selection_survives_navigation_and_source_reset():
    result = app()
    result.selectbox[0].select('rating').run()
    result.selectbox[1].select(None).run()
    result.sidebar.radio[0].set_value('About').run()
    result.sidebar.radio[0].set_value('Analyze Feedback').run()
    assert result.selectbox[0].value == 'rating'
    assert result.selectbox[1].value is None
    assert result.button[0].disabled
    result.main.radio[0].set_value('Upload CSV').run()
    assert result.session_state['intake']['prepared'] is None
    assert result.button[0].disabled
    result.main.radio[0].set_value('Sample dataset').run()
    assert result.selectbox[0].value == 'feedback'
    assert result.session_state['intake']['prepared'].quality['valid'] == 48
    assert not result.exception


class Upload(BytesIO):
    name = 'test.csv'


def test_uploaded_workflow_and_replacement_errors():
    result = app()
    with patch('src.ui.analysis.st.file_uploader', return_value=Upload(b'notes,date,score\nGreat!,2026-01-01,4\n  ,wrong,no\nok,,\n')):
        result.main.radio[0].set_value('Upload CSV').run()
        assert result.button[0].disabled  # No automatic guess for arbitrary text columns.
        result.selectbox[0].select('notes').run()
        result.selectbox[4].select('score').run()
        result.selectbox[5].select('date').run()
        prepared = result.session_state['intake']['prepared']
        assert prepared.quality['valid'] == 1
        assert prepared.quality['empty'] == 1
        assert prepared.quality['too_short'] == 1
        assert prepared.context_quality['feedback_date']['invalid'] == 1
        assert prepared.context_quality['rating']['invalid'] == 1
        assert not result.exception
        result.sidebar.radio[0].set_value('Overview').run()
    # Simulate Streamlit clearing the uploader widget when its page is absent.
    with patch('src.ui.analysis.st.file_uploader', return_value=None):
        result.sidebar.radio[0].set_value('Analyze Feedback').run()
        assert result.selectbox[0].value == 'notes'
        assert len(result.dataframe[0].value) == 3
        result.main.radio[0].set_value('Sample dataset').run()
        result.main.radio[0].set_value('Upload CSV').run()
        assert len(result.dataframe[0].value) == 3
        assert result.selectbox[0].value is None
    with patch('src.ui.analysis.st.file_uploader', return_value=Upload(b'a,a\n1,2')):
        result.run()
        assert result.error
        assert result.session_state['intake']['active_dataset'] is None
        assert result.session_state['intake']['prepared'] is None
        assert result.button[0].disabled
        assert not result.exception
