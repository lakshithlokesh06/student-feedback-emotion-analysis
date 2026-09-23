from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.config import PROJECT_ROOT
from tests.test_emotion import FakeModel


def test_dashboard_sections_filters_reset_and_navigation():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.radio[0].set_value('Analyze Feedback').run()
    model = FakeModel()
    with patch('src.ui.analysis.get_model', return_value=model) as loader:
        app.button[0].click().run()
        app.sidebar.radio[0].set_value('Emotion Dashboard').run()
        assert not app.exception
        for section in app.main.radio[0].options:
            app.main.radio[0].set_value(section).run()
            assert not app.exception, section
        app.multiselect[1].select('Computer Science').run()
        assert app.metric[0].value == '8'
        app.sidebar.radio[0].set_value('About').run()
        app.sidebar.radio[0].set_value('Emotion Dashboard').run()
        assert app.multiselect[1].value == ['Computer Science']
        assert app.metric[0].value == '8'
        app.selectbox[0].set_value('Low confidence').run()
        assert app.metric[0].value == '0'
        app.button[0].click().run()
        assert app.metric[0].value == '48'
        assert loader.call_count == 1
        assert not app.exception


def test_dashboard_without_optional_context():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    app.sidebar.radio[0].set_value('Analyze Feedback').run()
    for index in range(1, 6):
        app.selectbox[index].select(None).run()
    with patch('src.ui.analysis.get_model', return_value=FakeModel()):
        app.button[0].click().run()
    app.sidebar.radio[0].set_value('Emotion Dashboard').run()
    for section in ['Course / Subject / Semester', 'Rating vs Emotion', 'Trends Over Time']:
        app.main.radio[0].set_value(section).run()
        assert app.info
        assert not app.exception
