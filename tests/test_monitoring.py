import io
import numpy as np
import pandas as pd
import pytest
from src.config import MODEL_EMOTIONS
from src.data.loader import load_csv
from src.data.validation import DatasetValidationError
from src.monitoring.validation import validate_monitoring, from_main
from src.monitoring.distributions import distribution, jensen_shannon, ks_distance
from src.monitoring.drift import compare, severity
from src.monitoring.state import new_state, replace_dataset, snapshot
from src.monitoring.export import export_csv
from src.monitoring.temporal import temporal


def raw(labels=('joy','fear'), scores=(.8,.4)):
    return pd.DataFrame({'feedback':['Some useful feedback']*len(labels), 'emotion_label':labels, 'emotion_confidence':scores})


def view():
    return validate_monitoring(raw())


@pytest.mark.parametrize('column,value', [('emotion_label','happy'),('emotion_confidence','bad'),('emotion_confidence',-1),('emotion_confidence',1.1),('emotion_confidence',np.inf),('emotion_confidence',None),('feedback',''),('feedback',42)])
def test_invalid(column,value):
    data = raw(); data[column] = value
    with pytest.raises(DatasetValidationError): validate_monitoring(data)


@pytest.mark.parametrize('data', [pd.DataFrame(),raw().iloc[:0],raw().drop(columns='emotion_label'),pd.DataFrame([['abc','joy',.8,.8]],columns=['feedback','emotion_label','emotion_confidence','emotion_confidence'])])
def test_invalid_schema(data):
    with pytest.raises(DatasetValidationError): validate_monitoring(data)


@pytest.mark.parametrize('content', [b'feedback,emotion_label,emotion_confidence\n"broken', b'feedback,feedback\na,b', b'feedback,emotion_label\nx,y,z'])
def test_malformed(content):
    with pytest.raises(DatasetValidationError): load_csv(io.BytesIO(content),'test.csv')


def test_no_mutation_and_unclassified():
    data = raw(); data.loc[2] = [None,None,None]; original=data.copy(deep=True)
    frame=validate_monitoring(data)
    pd.testing.assert_frame_equal(data,original)
    report=compare(frame,frame)
    assert report['summaries'][0]['total']==3
    assert report['summaries'][0]['analyzed']==2
    assert report['details'].set_index('metric').loc['Missing feedback (%)','current_value']==pytest.approx(100/3)


def test_summary_and_changes():
    a=view(); b=validate_monitoring(raw(('fear','fear'),(.3,.5)))
    report=compare(a,b); table=report['details'].set_index('metric')
    assert report['summaries'][0]['average']==pytest.approx(.6)
    assert report['summaries'][1]['dominant']=='fear'
    assert table.loc['Confidence mean','change']==pytest.approx(-.2)
    assert table.loc['Confidence median','change']==pytest.approx(-.2)
    assert table.loc['Low-confidence rate (%)','change']==0
    assert list(report['emotions'].emotion)==list(MODEL_EMOTIONS)
    fear=report['emotions'].set_index('emotion').loc['fear']
    assert fear.percentage_point_change==50
    assert fear.current_count==2


@pytest.mark.parametrize('p,q,expected', [([1,0],[1,0],0),([1,0],[0,1],1),([1,1],[2,2],0)])
def test_js(p,q,expected): assert jensen_shannon(p,q)==pytest.approx(expected)


def test_js_difference():
    assert 0 < jensen_shannon([.5,.5],[.9,.1]) < 1
    assert jensen_shannon([.5,.5],[.9,.1])==pytest.approx(jensen_shannon([.9,.1],[.5,.5]))


@pytest.mark.parametrize('p,q', [([0,0],[1,0]),([-1,2],[1,1]),([np.nan,1],[1,1]),([1],[1,1])])
def test_js_invalid(p,q):
    with pytest.raises(ValueError): jensen_shannon(p,q)


@pytest.mark.parametrize('value,band', [(0,'minimal'),(.049,'minimal'),(.05,'noticeable'),(.15,'substantial'),(1,'substantial')])
def test_bands(value,band): assert severity(value)==band


def test_ks():
    assert ks_distance(pd.Series([0,0]),pd.Series([1,1]))==1
    assert ks_distance(pd.Series([.2,.5]),pd.Series([.5,.2]))==0


def test_lengths():
    a=raw(); a.feedback=['one two three','four five six']; b=a.copy(); b.feedback=['one two three four five six']*2
    report=compare(validate_monitoring(a),validate_monitoring(b))
    rows=report['details'].set_index('metric')
    assert rows.loc['word_count median','change']==3
    assert rows.loc['character_count mean','reference_value']==13
    assert 'word_count median' in report['signals'].metric.values


@pytest.mark.parametrize('column',['course','subject','semester'])
def test_context_union(column):
    a=raw(); b=raw(); a[column]=['Only reference','Shared']; b[column]=['Only current','Shared']
    table=compare(validate_monitoring(a),validate_monitoring(b))['contexts'][column].set_index('category')
    assert table.loc['Only reference','current_count']==0
    assert table.loc['Only current','reference_count']==0
    assert table.loc['Only current','percentage_point_change']==50


def test_rating_and_missing():
    a=raw();b=raw();a['rating']=[2,4];b['rating']=['',5];a['course']=['A','B'];b['course']=['',None]
    report=compare(validate_monitoring(a),validate_monitoring(b)); rows=report['details'].set_index('metric')
    assert rows.loc['Rating mean','change']==2
    assert rows.loc['Rating median','change']==2
    assert rows.loc['Missing course (%)','change']==100
    assert 'Missing subject (%)' not in rows.index


def test_ambiguity_margin():
    a=raw();a['confidence_margin']=[.05,.3]; b=a.copy();b.confidence_margin=[.02,.01]
    report=compare(validate_monitoring(a),validate_monitoring(b))
    assert report['details'].set_index('metric').loc['Ambiguous rate (%)','change']==50
    assert report['ambiguity'] is not None
    a.confidence_margin=['bad',.3]
    assert compare(validate_monitoring(a),view())['ambiguity'] is None


def test_full_scores_ambiguity():
    a=raw(('joy',),(.5,))
    for label in MODEL_EMOTIONS:a['score_'+label]=.49 if label=='fear' else .5 if label=='joy' else .002
    assert validate_monitoring(a).ambiguous_prediction.iloc[0]
    a.score_anger=9
    assert 'ambiguous_prediction' not in validate_monitoring(a)


def test_deterministic_signals_and_insights():
    a=view();b=validate_monitoring(raw(('anger','anger'),(.2,.2)))
    x,y=compare(a,b),compare(a,b)
    pd.testing.assert_frame_equal(x['signals'],y['signals'])
    assert x['insights']==y['insights']
    assert set(x['signals'].level)=={'review'}
    assert 'Confidence mean' in x['signals'].metric.values
    assert compare(a,a)['signals'].empty
    assert compare(a,a)['severity']=='minimal'


@pytest.mark.parametrize('key,columns',[('details',['metric','reference_value','current_value','change','monitoring_status']),('emotions',['emotion','reference_count','reference_percentage','current_count','current_percentage','percentage_point_change']),('signals',['level','signal','metric','observed','threshold','unit'])])
def test_exports(key,columns):
    report=compare(view(),view())
    frame=pd.read_csv(io.BytesIO(export_csv(report[key])))
    assert list(frame.columns)==columns


@pytest.mark.parametrize('role',['reference','current'])
def test_empty_comparison(role):
    args={'reference':view(),'current':view()};args[role]=view().iloc[:0]
    with pytest.raises(ValueError):compare(**args)


def test_one_row_optional_absent():
    a=validate_monitoring(raw(('joy',),(.8,)));report=compare(a,a)
    assert report['js']==0 and report['ks']==0
    assert report['contexts']=={}
    assert report['ambiguity'] is None
    assert report['low_by_emotion'].reference_percentage.isna().all()


def test_state_snapshot_and_isolation():
    state=new_state();data=raw();snapshot(state,data,.5)
    saved=state['reference'].copy();data.loc[0,'feedback']='Changed'
    state['results']={'old':True};state['filters']={'emotion':'joy'}
    replace_dataset(state,'current',data,'upload',.5)
    pd.testing.assert_frame_equal(saved,state['reference'])
    assert state['results'] is None and state['filters']=={}
    assert 'session' in state['sources']['reference']
    with pytest.raises(DatasetValidationError):replace_dataset(state,'current',pd.DataFrame(),'bad',.5)
    assert len(state['current'])==2


def test_temporal():
    a=raw();a['feedback_date']=['2026-01-01','2026-01-02'];a=validate_monitoring(a)
    for freq in ('Day','Week','Month'):
        comp,stats,excluded=temporal(a,freq)
        assert excluded==0 and stats.volume.sum()==2 and comp.Count.sum()==2
    assert temporal(a,'Month')[1].low_confidence_percentage.iloc[0]==50


def test_comparison_threshold_applies_to_by_emotion():
    a=validate_monitoring(raw(('joy',)*3,(.8,)*3))
    report=compare(a,a,threshold=.9)
    assert report['low_by_emotion'].set_index('emotion').loc['joy','current_percentage']==100


def test_temporal_gaps_remain_visible():
    a=raw();a['feedback_date']=['2026-01-01','2026-01-03']
    _,stats,_=temporal(validate_monitoring(a),'Day')
    assert stats.volume.tolist()==[1,0,1]
    assert pd.isna(stats.average_confidence.iloc[1])
