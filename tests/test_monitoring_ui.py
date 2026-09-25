from streamlit.testing.v1 import AppTest
from src.config import PROJECT_ROOT
from src.monitoring.state import new_state, replace_dataset
from tests.test_monitoring import raw


def test_monitoring_report_navigation_and_threshold():
    app=AppTest.from_file(str(PROJECT_ROOT/'app.py')).run()
    state=new_state()
    a=raw();a['feedback_date']=['2026-01-01','2026-01-02'];a['course']=['A','B']
    replace_dataset(state,'reference',a,'baseline',.5)
    replace_dataset(state,'current',a,'current',.5)
    app.session_state['monitoring']=state
    app.sidebar.radio[0].set_value('Model Monitoring').run()
    assert not app.exception
    assert len(app.get('download_button'))==3
    assert any('0.000'==x.value for x in app.metric)
    app.number_input[0].set_value(.9).run()
    assert not app.exception
    assert app.session_state['monitoring']['results']['summaries'][0]['low_percentage']==100
    for page in ('Analyze Feedback','Model Evaluation','Human Review','Model Monitoring'):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
    assert app.session_state['monitoring']['sources']['reference']=='baseline'
    assert len(app.session_state['monitoring']['reference'])==2
