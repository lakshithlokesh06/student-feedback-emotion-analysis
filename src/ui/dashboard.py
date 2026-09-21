import plotly.express as px
import streamlit as st

from src.emotion.classifier import summarize_results
from src.ui.results import render_summary


def render_dashboard() -> None:
    st.subheader("Emotion Dashboard")
    result = st.session_state.get('intake', {}).get('analysis')
    if result is None:
        st.info('No feedback has been analyzed yet. Prepare feedback and click Analyze Feedback to populate this dashboard.')
    else:
        render_summary(result)
        summary = summarize_results(result)
        left, right = st.columns(2)
        with left:
            st.markdown('#### Emotion distribution')
            distribution = summary['distribution']
            st.plotly_chart(px.bar(distribution, x='Emotion', y='Count', hover_data={'Percentage': ':.1f'}, color_discrete_sequence=['#167D8D']), width='stretch')
            st.dataframe(distribution, hide_index=True, width='stretch', column_config={'Percentage': st.column_config.NumberColumn(format='%.1f%%')})
        with right:
            st.markdown('#### Confidence distribution')
            scores = result.data.loc[result.data[result.fields['emotion_status']].eq('analyzed'), result.fields['emotion_confidence']]
            figure = px.histogram(x=scores, nbins=10, labels={'x': 'Confidence', 'y': 'Feedback count'}, color_discrete_sequence=['#167D8D'])
            figure.update_xaxes(range=[0, 1], tickformat='.0%')
            figure.add_vline(x=result.threshold, line_dash='dash')
            st.plotly_chart(figure, width='stretch')
        st.caption(f'Distributions include low-confidence predictions; the dashed line marks {result.threshold:.0%}. Percentages exclude unanalyzed rows.')
    st.caption('Planned: emotion trends, course/subject comparisons, recurring emotional patterns, and contextual review of high-concern feedback.')
