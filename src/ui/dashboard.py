import streamlit as st
from src.analytics.metrics import analytics_view, metrics, filtered_export
from src.analytics.filters import apply_filters
from src.emotion.classifier import export_csv
from src.ui.dashboard_filters import render_filters
from src.ui import dashboard_sections as sections


def render_dashboard() -> None:
    st.subheader('Emotion Dashboard')
    state = st.session_state.get('intake', {})
    result = state.get('analysis')
    if result is None:
        st.info('No feedback has been analyzed yet. Prepare feedback and click Analyze Feedback to populate this dashboard.')
        return
    preparation = state.get('prepared')
    view = analytics_view(result, preparation.fields if preparation else {}, state.get('contexts', {}))
    filters = render_filters(view, state.get('analytics_generation', 0), result.threshold)
    filtered = apply_filters(view, filters, result.threshold)
    st.caption(f'Current filtered view: {len(filtered):,} of {len(view):,} total rows. Model predictions are reused; filters never run inference.')
    stats = metrics(filtered, result.threshold)
    cards = [('Total Feedback', stats['total']), ('Feedback Analyzed', stats['analyzed']),
             ('Dominant Emotion', stats['dominant']), ('Average Confidence', f"{stats['average']:.0%}" if stats['average'] is not None else 'Unavailable'),
             ('Low-Confidence Predictions', stats['low']), ('Detected Categories', stats['categories'])]
    for start in (0, 3):
        for column, (title, value) in zip(st.columns(3), cards[start:start + 3]):
            column.metric(title, value)
    left, right = st.columns(2)
    left.download_button('Download full analyzed dataset', export_csv(result), 'analyzed_student_feedback.csv', 'text/csv')
    right.download_button('Download current filtered view', filtered_export(result, filtered), 'filtered_student_feedback.csv', 'text/csv')
    if not len(filtered):
        st.info('No rows match these filters. Reset Filters or broaden your selections.')
        return
    if not stats['analyzed']:
        st.info('This view has no analyzed feedback. Unusable rows remain available in the export.')
        return
    section = st.radio('Dashboard section', ['Overview', 'Course / Subject / Semester', 'Rating vs Emotion', 'Trends Over Time', 'Confidence Analysis', 'Feedback Review', 'Feedback Requiring Attention', 'Insights'], horizontal=True)
    if section == 'Overview':
        sections.overview(filtered, result.threshold)
    elif section == 'Course / Subject / Semester':
        sections.comparisons(filtered)
    elif section == 'Rating vs Emotion':
        sections.ratings(filtered)
    elif section == 'Trends Over Time':
        sections.trends(filtered)
    elif section == 'Confidence Analysis':
        sections.confidence(filtered, result.threshold)
    elif section == 'Feedback Review':
        sections.review(filtered, state)
    elif section == 'Feedback Requiring Attention':
        sections.review(filtered, state, attention=True)
    else:
        sections.insights(filtered, result.threshold)
