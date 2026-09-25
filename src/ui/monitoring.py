import streamlit as st
from src.config import LOW_CONFIDENCE_THRESHOLD
from src.data.loader import load_csv
from src.data.validation import DatasetValidationError
from src.monitoring.validation import from_main
from src.monitoring.state import new_state, replace_dataset, snapshot
from src.monitoring.drift import compare
from src.ui.monitoring_sections import report, limitations


def render_monitoring():
    st.subheader('Model Monitoring')
    st.write('Compare earlier reference feedback with current feedback. Data drift describes changes in inputs; prediction drift describes changes in emotion distributions; confidence drift describes changes in model scores. Drift does not automatically mean reduced accuracy.')
    st.session_state.setdefault('monitoring', new_state())
    state = st.session_state['monitoring']
    parent = st.session_state.get('intake', {})
    st.caption('Snapshots and uploads last for this session only. To use bundled synthetic data, classify it in Analyze Feedback and capture a reference snapshot here. Then analyze a different CSV and load it as current. Raw uploads require analysis in that existing workflow; monitoring never runs inference.')
    threshold = st.number_input('Monitoring low-confidence threshold', 0.0, 1.0, value=float(state.get('threshold', parent['analysis'].threshold if parent.get('analysis') is not None else LOW_CONFIDENCE_THRESHOLD)), step=.05, key='_monitor_threshold')
    if state.get('threshold') != threshold:
        state['threshold'] = threshold
        for role in ('reference','current'):
            if state[role] is not None:
                state[role]['emotion_low_confidence'] = state[role].emotion_confidence.lt(threshold)
        state['results'] = None
    st.caption('Both datasets use this shared threshold; uploaded low-confidence flags are recomputed. The default follows the main classifier threshold.')
    for container, role in zip(st.columns(2), ('reference','current')):
        with container:
            st.markdown('#### ' + role.title() + ' dataset')
            label = 'Set main analysis as Reference Snapshot' if role == 'reference' else 'Load main analysis as Current'
            if st.button(label, disabled=parent.get('analysis') is None):
                try:
                    frame = from_main(parent)
                    if role == 'reference':
                        snapshot(state, frame, threshold)
                    else:
                        replace_dataset(state, role, frame, 'Main analysis snapshot', threshold)
                    state.pop(role+'_error', None)
                except DatasetValidationError as exc:
                    state[role+'_error'] = str(exc)
            upload = st.file_uploader('Upload analyzed ' + role + ' CSV', type=['csv'], key='_monitor_upload_'+role)
            if st.button('Use uploaded ' + role, disabled=upload is None):
                try:
                    replace_dataset(state, role, load_csv(upload, upload.name), upload.name, threshold)
                    state.pop(role+'_error', None)
                except DatasetValidationError as exc:
                    state[role+'_error'] = str(exc)
            if state.get(role+'_error'):
                st.error(state[role+'_error'] + ' The previous valid snapshot, if any, remains selected.')
            if state[role] is not None:
                st.caption(f"Selected: {state['sources'][role]} · {len(state[role]):,} rows")
    if any(state[role] is None for role in ('reference','current')):
        st.info('Select a reference and a current analyzed dataset to build the monitoring report. Uploaded CSVs require feedback, emotion_label, and emotion_confidence.')
        limitations()
        return
    if state['results'] is None:
        state['results'] = compare(state['reference'], state['current'], threshold)
    report(state)
