from streamlit.testing.v1 import AppTest

from src.config import NAVIGATION_LABELS, PROJECT_ROOT


def test_navigation_and_analysis_placeholder():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run()
    assert not app.exception
    assert app.title[0].value == 'Student Feedback Emotion Analysis'
    for page in NAVIGATION_LABELS:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        if page == 'Analyze Feedback':
            assert len(app.dataframe[0].value) == 48
            app.button[0].click().run()
            assert any('No predictions' in item.value for item in app.info)
            app.selectbox[0].select('rating').run()
            assert app.button[0].disabled
            app.main.radio[0].set_value('Upload CSV').run()
            assert app.button[0].disabled
