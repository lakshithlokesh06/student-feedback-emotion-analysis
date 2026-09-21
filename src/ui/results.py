"""Shared presentation of real predictions."""
import streamlit as st

from src.config import MAX_MODEL_TOKENS, MODEL_EMOTIONS, MODEL_NAME, PREVIEW_ROWS
from src.emotion.classifier import export_csv, summarize_results


def render_model_info(metadata: dict | None = None) -> None:
    metadata = metadata or {}
    with st.expander('Model information and limitations'):
        st.markdown(f"**Model:** [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})")
        st.write('Provider: Jochen Hartmann via Hugging Face · Task: English emotion text classification · CPU inference')
        st.write('Supported labels: ' + ', '.join(metadata.get('labels', MODEL_EMOTIONS)))
        st.write(f"Inputs are truncated to {metadata.get('max_tokens', MAX_MODEL_TOKENS)} tokens including special tokens. Original feedback remains intact.")
        if metadata.get('revision'):
            st.caption('Loaded revision: ' + metadata['revision'])
        st.write('Confidence is the top-class softmax probability, not a calibrated probability of correctness. Predictions may be wrong, particularly for sarcasm, mixed emotions, and non-English or unfamiliar text. This model has not been evaluated on this project’s student feedback.')
        st.write('Frustration and satisfaction cannot be reliably derived from the native labels and are not predicted. Use results for educational exploration and human review, not as ground truth or an assessment of a student’s mental state.')


def render_summary(result) -> None:
    summary = summarize_results(result)
    metrics = [('Total Feedback', summary['total']), ('Feedback Analyzed', summary['analyzed']),
               ('Feedback Not Analyzed', summary['not_analyzed']), ('Dominant Emotion', summary['dominant']),
               ('Categories Detected', summary['categories']),
               ('Average Confidence', f"{summary['average_confidence']:.0%}" if summary['average_confidence'] is not None else 'Unavailable')]
    for start in (0, 3):
        for column, (label, value) in zip(st.columns(3), metrics[start:start + 3]):
            column.metric(label, value)
    st.caption('Metrics use analyzed rows only where applicable. Tied dominant emotions are listed together.')


def render_results(state: dict) -> None:
    result = state.get('analysis')
    if result is None:
        return
    st.subheader('Classification results')
    render_summary(result)
    st.info(f'Predictions below {result.threshold:.0%} confidence are flagged for cautious interpretation and remain included in results.')
    label_field = result.fields['emotion_label']
    emotions = sorted(result.data[label_field].dropna().unique())
    selected = st.selectbox('Filter by emotion', ['All rows', *emotions], key='emotion_filter')
    frame = result.data if selected == 'All rows' else result.data.loc[result.data[label_field].eq(selected)]
    columns = list(dict.fromkeys([state['feedback_column'], *(c for c in state['contexts'].values() if c), *result.fields.values()]))
    preview = frame[columns].head(PREVIEW_ROWS).copy()
    confidence = result.fields['emotion_confidence']
    preview[confidence] = preview[confidence].map(lambda score: f'{score:.0%}' if score == score else '—')
    st.caption(f'{len(frame):,} matching rows · Showing up to {PREVIEW_ROWS} · CSV retains numeric confidence values')
    st.dataframe(preview, hide_index=True, width='stretch')
    st.download_button('Download analyzed feedback CSV', export_csv(result), file_name='analyzed_student_feedback.csv', mime='text/csv')
    render_model_info(result.metadata)
