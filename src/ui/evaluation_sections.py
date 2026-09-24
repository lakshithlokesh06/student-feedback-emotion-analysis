import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MODEL_EMOTIONS, PREVIEW_ROWS
from src.evaluation.metrics import performance, confusion, class_distribution
from src.evaluation.confidence import confidence_groups, correctness_confidence, calibration, truncation_summary
from src.evaluation.errors import filter_errors, high_confidence_errors
from src.evaluation.runner import evaluated, export_results
from src.evaluation.quality import quality_warnings


def chart(figure):
    st.plotly_chart(figure, width='stretch', config={'modeBarButtonsToRemove': ['toImage']})


def summary(result, synthetic):
    view = result.view()
    rows = evaluated(view)
    stats, classes = performance(view)
    count = len(rows)
    support = classes.Support.max()
    highest = ', '.join(classes.loc[classes.Support.eq(support), 'Emotion']) if support else 'Unavailable'
    cards = [('Evaluation Rows', len(view)), ('Successfully Evaluated', count), ('Rejected', len(view) - count),
             ('Accuracy', f"{stats['accuracy']:.1%}" if count else 'Unavailable'),
             ('Macro F1', f"{stats['macro_f1']:.1%}" if count else 'Unavailable'),
             ('Weighted F1', f"{stats['weighted_f1']:.1%}" if count else 'Unavailable')]
    for start in (0, 3):
        for column, (label, value) in zip(st.columns(3), cards[start:start+3]):
            column.metric(label, value)
    st.write(f"Highest-support emotion(s): {highest} · Incorrect: {int(rows.is_correct.eq(False).sum())} · High-confidence misclassifications: {len(high_confidence_errors(view, result.high_confidence_threshold))} · Ambiguous: {int(rows.ambiguous_prediction.eq(True).sum())}")
    for message in quality_warnings(view, synthetic, result.duplicate_count):
        st.warning(message)
    for column, label, raw, filename in zip(st.columns(3),
            ['Evaluation Results CSV', 'Per-Class Metrics CSV', 'Misclassifications CSV'],
            [export_results(result), classes.to_csv(index=False).encode('utf-8'), export_results(result, rows.loc[rows.is_correct.eq(False)].index)],
            ['evaluation_results.csv', 'evaluation_metrics.csv', 'evaluation_misclassifications.csv']):
        column.download_button(label, raw, filename, 'text/csv')
    return view, stats, classes


def performance_section(view, stats, classes):
    st.dataframe({'Metric': list(stats), 'Value': list(stats.values())}, hide_index=True, column_config={'Value': st.column_config.NumberColumn(format='%.3f')})
    st.caption('Macro metrics average all seven native classes, including absent classes scored as zero; weighted metrics use true-label support. Undefined class precision/recall is set to zero. Inspect support before interpreting either average.')
    st.markdown('#### Per-emotion performance')
    st.dataframe(classes, hide_index=True, width='stretch', column_config={name: st.column_config.NumberColumn(format='%.3f') for name in ('Precision','Recall','F1')})
    st.caption('Support is the number of evaluated reference examples. Fewer than three examples provides very weak evidence; zero support does not measure recall for that emotion.')
    normalized = st.checkbox('Normalize confusion matrix by true emotion')
    matrix = confusion(view, normalized)
    figure = px.imshow(matrix, text_auto='.1f' if normalized else True, labels={'x': 'Predicted emotion', 'y': 'True emotion', 'color': 'Percent' if normalized else 'Count'}, color_continuous_scale='Blues', aspect='auto')
    chart(figure)
    st.caption('Rows are reference labels and columns are predictions. Diagonal cells are matches. Empty true-label rows have zero counts (displayed as zero when normalized).')
    distribution = class_distribution(view)
    chart(px.bar(distribution, x='Emotion', y='Count', color='Labels', barmode='group', color_discrete_sequence=['#167D8D','#80969F'], hover_data={'Percentage': ':.1f'}, category_orders={'Emotion': list(MODEL_EMOTIONS)}))
    st.dataframe(distribution, hide_index=True, width='stretch', column_config={'Percentage': st.column_config.NumberColumn(format='%.1f%%')})


def confidence_section(view):
    rows = evaluated(view)
    st.markdown('#### Confidence versus correctness')
    st.dataframe(correctness_confidence(view), hide_index=True)
    frame = rows.assign(Correctness=rows.is_correct.map({True: 'Correct', False: 'Incorrect'}))
    chart(px.histogram(frame, x='emotion_confidence', color='Correctness', barmode='overlay', nbins=10, opacity=.7, color_discrete_sequence=['#167D8D','#80969F']))
    st.dataframe(confidence_groups(view), hide_index=True)
    st.caption('Bands are left-inclusive and right-exclusive, except the final band includes 1.00. Empty bands have no accuracy estimate.')
    st.markdown('#### Reliability / calibration')
    bins = calibration(view)
    occupied = bins.loc[bins.Count.gt(0)]
    figure = go.Figure()
    figure.add_scatter(x=[0,1], y=[0,1], mode='lines', name='Reference: confidence equals accuracy', line={'dash':'dash','color':'#A0AAAE'})
    figure.add_scatter(x=occupied.Mean_confidence, y=occupied.Accuracy, mode='markers', name='Observed bins', text=occupied.Count.map(lambda n: f'{n} examples'), marker={'size':10,'color':'#167D8D'})
    figure.update_layout(xaxis_title='Mean confidence', yaxis_title='Observed accuracy', xaxis_range=[0,1], yaxis_range=[0,1])
    chart(figure)
    st.dataframe(bins, hide_index=True)
    st.info('These are descriptive reliability estimates, not validated calibration. Small bins are unstable. High-confidence predictions can still be incorrect; no recalibration is performed.')
    st.markdown('#### Truncation')
    truncated = int(rows.was_truncated.eq(True).sum())
    st.write(f'{truncated} of {len(rows)} evaluated rows were truncated ({truncated / len(rows):.1%}).')
    comparison = truncation_summary(view)
    if len(comparison) == 2 and comparison.Count.min() >= 3:
        st.dataframe(comparison, hide_index=True)
    else:
        st.caption('At least three rows in each truncation group are required to display a performance comparison.')
    st.caption('This comparison cannot establish that truncation caused an error.')


def _save_filter(name):
    st.session_state['evaluation']['filters'][name] = st.session_state['_eval_filter_' + name]


def review_table(result, rows):
    columns = list(dict.fromkeys(['feedback', 'true_emotion', *result.fields.values(), *(c for c in ('course','subject','semester','rating','feedback_date') if c in result.analysis.data)]))
    # Keep score vectors out of row tables; they have a dedicated transparency view.
    columns = [c for c in columns if c not in [result.fields[f'score_{label}'] for label in MODEL_EMOTIONS]]
    st.caption(f'{len(rows)} matching rows · Showing up to {PREVIEW_ROWS}')
    st.dataframe(result.analysis.data.iloc[rows.index[:PREVIEW_ROWS]][columns], hide_index=True, width='stretch')


def errors_section(result, view, state):
    filters = state['filters']
    for name in ('true', 'predicted'):
        st.session_state['_eval_filter_' + name] = filters[name]
        st.multiselect(f'{name.title()} emotion', list(MODEL_EMOTIONS), key='_eval_filter_' + name, on_change=_save_filter, args=(name,))
    st.session_state['_eval_filter_correctness'] = filters['correctness']
    st.selectbox('Correct / incorrect', ['All','Correct','Incorrect'], key='_eval_filter_correctness', on_change=_save_filter, args=('correctness',))
    st.session_state['_eval_filter_confidence'] = filters['confidence']
    st.slider('Minimum prediction confidence', 0.0, 1.0, key='_eval_filter_confidence', on_change=_save_filter, args=('confidence',))
    rows = filter_errors(view, filters['true'], filters['predicted'], filters['correctness'], filters['confidence'])
    st.caption('These filters affect review tables only; headline metrics and exports remain based on the full evaluation.')
    review_table(result, rows)
    st.markdown('#### High-confidence misclassifications')
    st.caption(f'Incorrect predictions at or above {result.high_confidence_threshold:.0%} confidence, within the review filters.')
    review_table(result, high_confidence_errors(rows, result.high_confidence_threshold))
    st.markdown('#### Ambiguous predictions')
    st.caption(f'Top-two score margin below {result.ambiguity_threshold:.0%}, within the review filters. This is an analytical flag, not a claim about understanding.')
    review_table(result, rows.loc[rows.ambiguous_prediction.eq(True)])


def transparency_section(result, view):
    rows = evaluated(view)
    position = st.selectbox('Evaluation row', rows.index.tolist(), format_func=lambda i: f'Row {i + 1}: {str(result.analysis.data.iloc[i]["feedback"])[:80]}')
    row = rows.loc[position]
    st.write(result.analysis.data.iloc[position]['feedback'])
    st.write(f'True label: {row.true_emotion_normalized} · Prediction: {row.predicted_emotion} · Confidence: {row.emotion_confidence:.1%}')
    st.write(f'Second emotion: {row.second_emotion} ({row.second_emotion_confidence:.1%}) · Margin: {row.confidence_margin:.1%}')
    st.write(f'Tokenized length including special tokens: {int(row.token_length)} · Truncated: {bool(row.was_truncated)} · Limit: {result.analysis.metadata.get("max_tokens", "Unavailable")} tokens')
    scores = {'Emotion': list(MODEL_EMOTIONS), 'Probability': [row[f'score_{label}'] for label in MODEL_EMOTIONS]}
    chart(px.bar(scores, x='Emotion', y='Probability', color_discrete_sequence=['#167D8D']))
    st.info('This view exposes model scores and truncation, not word-level importance or causal explanations. No token-attribution method is implemented.')
