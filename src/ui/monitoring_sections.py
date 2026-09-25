import pandas as pd
import plotly.express as px
import streamlit as st
from src.config import MODEL_EMOTIONS, MONITORING_THRESHOLDS, AMBIGUITY_MARGIN_THRESHOLD
from src.analytics.metrics import analyzed
from src.monitoring.drift import missing
from src.monitoring.export import export_csv
from src.monitoring.temporal import temporal


def limitations():
    st.markdown('#### Limitations')
    st.caption('Drift does not prove accuracy degradation. Prediction changes may reflect genuine changes in student feedback. Small datasets produce unstable estimates. Confidence is not calibrated accuracy. Drift severity thresholds are heuristic monitoring aids, not universal statistical guarantees. Meaningful performance monitoring requires labeled data and Model Evaluation. Human Review captures annotations; monitoring identifies changes. This is session-based descriptive monitoring, not production monitoring infrastructure.')


def show(table):
    st.dataframe(table, hide_index=True, width='stretch', height=min(360, 38+35*len(table)))


def distribution_chart(table, category, title):
    view = table.melt(id_vars=category, value_vars=['reference_percentage','current_percentage'], var_name='Dataset', value_name='Percentage')
    view['Dataset'] = view.Dataset.map({'reference_percentage': 'Reference', 'current_percentage': 'Current'})
    st.plotly_chart(px.bar(view, x=category, y='Percentage', color='Dataset', barmode='group', title=title), width='stretch')


def report(state):
    result, reference, current = state['results'], state['reference'], state['current']
    a, b = analyzed(reference), analyzed(current)
    st.markdown('#### Dataset summary')
    for container, name, summary in zip(st.columns(2), ('Reference','Current'), result['summaries']):
        with container:
            st.markdown('**'+name+'**')
            cols = st.columns(2)
            cols[0].metric('Feedback rows', summary['total'])
            cols[1].metric('Analyzed rows', summary['analyzed'])
            st.write('Dominant emotion: ' + summary['dominant'])
            cols = st.columns(2)
            cols[0].metric('Average confidence', f"{summary['average']:.1%}")
            cols[1].metric('Low-confidence share', f"{summary['low_percentage']:.1f}%")
    s0, s1 = result['summaries']
    st.caption(f"Feedback volume: {100*(s1['total']/s0['total']-1):+.1f}% · Mean confidence: {100*(s1['average']-s0['average']):+.1f} percentage points · Low-confidence share: {s1['low_percentage']-s0['low_percentage']:+.1f} percentage points")
    if min(len(a),len(b)) < 30:
        st.info('At least one dataset has fewer than 30 analyzed rows. Drift estimates may be unstable.')
    st.markdown('#### Emotion drift')
    st.metric('Jensen–Shannon divergence', f"{result['js']:.3f}")
    st.caption(f"{result['severity'].title()} difference. Base-2 Jensen–Shannon divergence compares emotion proportions: 0 = identical distributions, 1 = disjoint distributions; larger values mean greater difference. It does not measure accuracy.")
    st.caption(f"Heuristic bands: minimal < {MONITORING_THRESHOLDS['js_noticeable']}; noticeable < {MONITORING_THRESHOLDS['js_substantial']}; substantial ≥ {MONITORING_THRESHOLDS['js_substantial']}. Thresholds are monitoring aids, not universal statistical guarantees.")
    distribution_chart(result['emotions'], 'emotion', 'Emotion prediction shares')
    show(result['emotions'].round(2))
    st.markdown('#### Confidence drift')
    confidence = pd.concat([a[['emotion_confidence']].assign(Dataset='Reference'), b[['emotion_confidence']].assign(Dataset='Current')], ignore_index=True)
    st.plotly_chart(px.histogram(confidence, x='emotion_confidence', color='Dataset', barmode='overlay', histnorm='percent', nbins=20, opacity=.6, range_x=[0,1]), width='stretch')
    st.caption(f"Empirical KS statistic: {result['ks']:.3f}. Maximum difference between confidence cumulative distributions, from 0 to 1. This is a distribution effect size; no p-value or significance claim is made.")
    show(result['details'].loc[result['details'].metric.str.startswith(('Confidence','Low-confidence'))].round(3))
    st.caption(f"Low-confidence rates by emotion require at least {MONITORING_THRESHOLDS['minimum_emotion_support']} predictions per dataset/emotion; otherwise the rate is unavailable. Higher low-confidence rates do not establish degradation.")
    show(result['low_by_emotion'].round(2))
    st.markdown('#### Text-length drift')
    st.caption('Character and whitespace-delimited word counts use cleaned usable feedback. Missing, non-text, and too-short feedback are excluded from length statistics. Longer feedback is not necessarily better.')
    show(result['details'].loc[result['details'].metric.str.startswith(('character_count','word_count'))].round(2))
    lengths = pd.concat([reference[['character_count','word_count']].assign(Dataset='Reference'), current[['character_count','word_count']].assign(Dataset='Current')])
    field = st.selectbox('Length distribution', ['word_count','character_count'])
    st.plotly_chart(px.histogram(lengths, x=field, color='Dataset', barmode='overlay', histnorm='percent', opacity=.6, nbins=25), width='stretch')
    st.markdown('#### Context and missing-data drift')
    st.caption('Context distributions use nonmissing values; ratings use valid numeric values. Missing-data rates use all rows. Unavailable fields are omitted. Invalid ratings/dates are excluded from their numerical/temporal views, not counted as missing.')
    if result['contexts']:
        for column, table in result['contexts'].items():
            with st.expander(column.title() + ' distribution'):
                distribution_chart(table, 'category', column.title()+' composition')
                show(table.round(2))
        show(result['details'].loc[result['details'].metric.str.startswith('Rating')].round(2))
    else:
        st.info('No shared optional context columns are available.')
    show(result['details'].loc[result['details'].metric.str.startswith('Missing ')].round(2))
    st.markdown('#### Ambiguity drift')
    if result['ambiguity'] is None:
        st.info('Ambiguity comparison unavailable: both datasets need valid full class scores or confidence margins. Low confidence is not used as an ambiguity proxy.')
    else:
        st.caption(f'Phase 5 definition: top-two confidence margin < {AMBIGUITY_MARGIN_THRESHOLD}. By-emotion rates use the same minimum support as low-confidence rates.')
        show(result['details'].loc[result['details'].metric.str.startswith('Ambiguous')].round(2))
        show(result['ambiguity'].round(2))
    with st.expander('Temporal monitoring'):
        if all('date' in frame and frame.date.notna().any() for frame in (reference,current)):
            frequency = st.selectbox('Time aggregation', ['Day','Week','Month'])
            try:
                stats, composition = [], []
                for name, frame in [('Reference',reference),('Current',current)]:
                    comp, summary, excluded = temporal(frame,frequency)
                    st.caption(f'{name}: {excluded} analyzed rows excluded for unavailable/invalid dates. Periods use UTC; weeks start Monday.')
                    if not summary.empty:
                        stats.append(summary.assign(Dataset=name)); composition.append(comp.assign(Dataset=name))
                if stats:
                    combined = pd.concat(stats)
                    for metric in ('volume','average_confidence','low_confidence_percentage'):
                        st.plotly_chart(px.line(combined,x='Period',y=metric,color='Dataset',markers=True),width='stretch')
                    st.plotly_chart(px.line(pd.concat(composition),x='Period',y='Percentage',color='Emotion',facet_row='Dataset'),width='stretch')
            except ValueError as exc:
                st.info(str(exc))
        else:
            st.info('Valid feedback_date values in both datasets are required for temporal monitoring.')
    with st.expander('Emotion-specific inspection', expanded=False):
        selected = st.selectbox('Emotion to inspect', MODEL_EMOTIONS, index=list(MODEL_EMOTIONS).index(state['filters'].get('emotion','anger')))
        state['filters']['emotion'] = selected
        show(result['emotions'].loc[result['emotions'].emotion.eq(selected)].round(2))
        show(pd.DataFrame([{'Dataset': name, 'Average confidence': frame.loc[frame.emotion_label.eq(selected),'emotion_confidence'].mean()} for name,frame in [('Reference',a),('Current',b)]]).round(3))
    st.markdown('#### Monitoring signals')
    if result['signals'].empty:
        st.info('No configured monitoring signal detected. This does not establish model accuracy.')
    else:
        show(result['signals'].round(3))
    with st.expander('Configured heuristic rules'):
        st.json(MONITORING_THRESHOLDS)
    st.markdown('#### Deterministic monitoring summary')
    for insight in result['insights']:
        st.write('• ' + insight)
    with st.expander('Monitoring details and current-dataset drilldown'):
        show(result['details'].round(3))
        modes = ['Low-confidence predictions','Selected emotion','Longest feedback','Missing context']
        if 'ambiguous_prediction' in current:
            modes.append('Ambiguous predictions')
        mode = st.selectbox('Inspect current rows', modes, index=modes.index(state['filters'].get('drilldown',modes[0])) if state['filters'].get('drilldown',modes[0]) in modes else 0)
        state['filters']['drilldown'] = mode
        if mode == modes[0]:
            rows = b.loc[b.emotion_low_confidence]
        elif mode == 'Selected emotion':
            rows = b.loc[b.emotion_label.eq(selected)]
        elif mode == 'Longest feedback':
            rows = current.sort_values('word_count',ascending=False)
        elif mode == 'Ambiguous predictions':
            rows = b.loc[b.ambiguous_prediction]
        else:
            columns = [c for c in ('course','subject','semester','rating','feedback_date') if c in current]
            mask = pd.concat([missing(current[c]) for c in columns],axis=1).any(axis=1) if columns else pd.Series(False,index=current.index)
            rows = current.loc[mask]
        st.caption(f'{len(rows)} matching rows; showing at most 50. Inspection only.')
        show(rows.head(50))
    st.markdown('#### Monitoring exports')
    for label, key, filename in [('Monitoring Summary CSV','details','monitoring_summary.csv'),('Emotion Drift CSV','emotions','emotion_drift.csv'),('Monitoring Signals CSV','signals','monitoring_signals.csv')]:
        st.download_button(label, export_csv(result[key]), filename, 'text/csv')
    limitations()
