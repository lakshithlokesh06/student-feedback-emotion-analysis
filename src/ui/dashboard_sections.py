"""Small renderers for the dashboard's optional analytical sections."""
import plotly.express as px
import streamlit as st

from src.config import MODEL_EMOTIONS, PREVIEW_ROWS
from src.analytics.metrics import analyzed, metrics
from src.analytics.distributions import emotion_distribution, confidence_by_emotion
from src.analytics.comparisons import group_comparison, rating_comparison
from src.analytics.trends import default_frequency, temporal_distribution
from src.analytics.insights import attention_feedback, descriptive_insights

COLORS = ['#315B70', '#597680', '#748C96', '#237F85', '#9AA9AB', '#5C6E91', '#88AAAE']
ORDER = {'Emotion': list(MODEL_EMOTIONS), 'emotion_label': list(MODEL_EMOTIONS)}


def chart(figure):
    st.plotly_chart(figure, width='stretch', config={'modeBarButtonsToRemove': ['toImage']})


def overview(data, threshold):
    left, right = st.columns(2)
    with left:
        st.markdown('#### Emotion distribution')
        distribution = emotion_distribution(data)
        chart(px.bar(distribution, x='Emotion', y='Count', hover_data={'Percentage': ':.1f'}, color_discrete_sequence=['#167D8D'], category_orders=ORDER))
        st.dataframe(distribution, hide_index=True, width='stretch', column_config={'Percentage': st.column_config.NumberColumn(format='%.1f%%')})
    with right:
        st.markdown('#### Confidence distribution')
        figure = px.histogram(analyzed(data), x='emotion_confidence', nbins=10, labels={'emotion_confidence': 'Prediction confidence'}, color_discrete_sequence=['#167D8D'])
        figure.update_xaxes(range=[0, 1], tickformat='.0%')
        figure.add_vline(x=threshold, line_dash='dash')
        chart(figure)
        st.caption('Percentages use analyzed rows only. The dashed line marks the low-confidence threshold.')


def confidence(data, threshold):
    stats = metrics(data, threshold)
    st.write(f"Median confidence: {stats['median']:.0%} · Low-confidence predictions: {stats['low']} ({stats['low_percentage']:.1f}%)")
    chart(px.box(analyzed(data), x='emotion_label', y='emotion_confidence', category_orders=ORDER, labels={'emotion_label': 'Emotion', 'emotion_confidence': 'Prediction confidence'}, color_discrete_sequence=['#167D8D']))
    st.dataframe(confidence_by_emotion(data), hide_index=True, width='stretch')
    st.info('Model confidence is not validated accuracy. High confidence does not guarantee correctness.')


def comparisons(data):
    for role in ('course', 'subject', 'semester'):
        st.markdown(f'#### {role.title()} analysis')
        summary, composition, total = group_comparison(data, role)
        if summary.empty:
            st.info(f'No usable {role} context is available in this view. Select a {role} column during intake, or adjust filters.')
            continue
        missing = int(analyzed(data)[role].isna().sum())
        st.caption(f'Showing {len(summary)} of {total} groups by feedback volume; use dashboard filters to select other groups. {missing} analyzed rows lack this context and are excluded.')
        st.dataframe(summary, hide_index=True, width='stretch', column_config={'Average confidence': st.column_config.NumberColumn(format='%.2f')})
        chart(px.bar(composition, x='Group', y='Count', color='Emotion', barmode='stack', category_orders=ORDER,
                     color_discrete_sequence=COLORS, hover_data={'Percentage': ':.1f'}))
        with st.expander(f'{role.title()} emotion percentages'):
            st.dataframe(composition, hide_index=True, width='stretch')
    st.caption('Group summaries describe predictions; they do not rank courses or subjects as best or worst.')


def ratings(data):
    averages, distribution, excluded = rating_comparison(data)
    if averages.empty:
        st.info('No valid numeric ratings are available. Select a rating column during intake or adjust filters.')
        return
    st.caption(f'{excluded} analyzed rows with missing/invalid ratings are excluded from this section only.')
    st.markdown('#### Average rating by emotion')
    chart(px.bar(averages, x='emotion_label', y='mean', hover_data=['count'], labels={'emotion_label': 'Emotion', 'mean': 'Average rating'}, category_orders=ORDER, color_discrete_sequence=['#167D8D']))
    st.dataframe(averages, hide_index=True)
    counts = distribution.groupby('rating').Count.sum().sort_values(ascending=False, kind='stable')
    selected = counts.head(20).index
    st.caption(f'Showing {len(selected)} of {len(counts)} rating values by volume. Use the rating filter to inspect other values.')
    chart(px.bar(distribution.loc[distribution.rating.isin(selected)], x='rating', y='Count', color='emotion_label',
                 category_orders=ORDER, color_discrete_sequence=COLORS, hover_data={'Percentage': ':.1f'}))
    st.dataframe(counts.head(20).rename('Feedback count').reset_index(), hide_index=True)
    st.info('These comparisons show association, not causation. No rating scale is assumed.')


def trends(data):
    if 'date' not in data:
        st.info('Select a feedback date column during intake to view trends.')
        return
    options = ['Day', 'Week', 'Month']
    frequency = st.selectbox('Time aggregation', options, index=options.index(default_frequency(data)))
    try:
        distribution, excluded = temporal_distribution(data, frequency)
    except ValueError as exc:
        st.info(str(exc))
        return
    st.caption(f'{excluded} analyzed rows excluded from trends due to missing/invalid dates. Original rows remain intact. Weeks start Monday; all dates are interpreted in UTC.')
    if distribution.empty:
        st.info('No valid dates are available for the currently analyzed feedback.')
        return
    volume = distribution.groupby('Period', as_index=False).Count.sum()
    chart(px.line(volume, x='Period', y='Count', markers=True, color_discrete_sequence=['#167D8D'], title='Feedback volume over time'))
    measure = st.radio('Emotion trend measure', ['Count', 'Percentage'], horizontal=True)
    chart(px.line(distribution, x='Period', y=measure, color='Emotion', category_orders=ORDER, color_discrete_sequence=COLORS, markers=True))
    st.caption('Unobserved periods have zero volume; percentages are undefined in those periods. Sparse periods may give unstable proportions.')


def review(data, state, attention=False):
    if attention:
        st.markdown('#### Feedback that may benefit from closer review')
        view = attention_feedback(data)
        st.info('This review aid selects anger, fear, sadness, and disgust predictions, ordered by confidence. It does not diagnose wellbeing, assess danger, or recommend action. A human must review predictions and context before any decision.')
    else:
        st.caption('Use the dashboard emotion, confidence, course, and subject filters to narrow this table.')
        ascending = st.radio('Sort confidence', ['Lowest first', 'Highest first'], horizontal=True) == 'Lowest first'
        view = analyzed(data).sort_values('emotion_confidence', ascending=ascending, kind='stable')
    if view.empty:
        st.info('No feedback matches this review view.')
        return
    result = state['analysis']
    columns = list(dict.fromkeys([state['feedback_column'], *result.fields.values(), *(v for v in state['contexts'].values() if v in result.data)]))
    frame = result.data.iloc[view.index[:PREVIEW_ROWS]][columns].copy()
    field = result.fields['emotion_confidence']
    frame[field] = frame[field].map(lambda value: f'{value:.0%}')
    st.caption(f'{len(view)} matching analyzed responses · Showing up to {PREVIEW_ROWS}')
    st.dataframe(frame, hide_index=True, width='stretch')


def insights(data, threshold):
    for observation in descriptive_insights(data, threshold):
        st.write('• ' + observation)
