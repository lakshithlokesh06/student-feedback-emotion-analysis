import hashlib

import streamlit as st

from src.config import CONTEXT_COLUMNS, MAX_COLUMNS, MAX_FILE_SIZE_BYTES, MAX_ROWS, MIN_FEEDBACK_LENGTH, PREVIEW_ROWS
from src.data.loader import load_csv
from src.data.preparation import prepare_feedback
from src.data.profiling import profile_dataset
from src.data.sample_data import load_sample_data
from src.data.validation import DatasetValidationError
from src.ui.intake_state import reset_dataset, invalidate_analysis
from src.ui.model_cache import get_model
from src.ui.results import render_results
from src.emotion.classifier import classify_feedback
from src.emotion.labels import EmotionError


def _save_widget(field: str, role: str | None = None) -> None:
    state = st.session_state['intake']
    if role is None:
        state[field] = st.session_state[f'_intake_{field}']
    else:
        state['contexts'][role] = st.session_state[f'_intake_{role}']


def _selection(label: str, options: list, value, field: str, role: str | None = None):
    key = f'_intake_{role or field}'
    st.session_state[key] = value
    return st.selectbox(label, options, key=key, format_func=lambda item: 'Not selected' if item is None else item,
                        on_change=_save_widget, args=(field, role))


def render_analysis() -> None:
    st.subheader("Prepare your feedback")
    st.write("Choose a dataset, inspect its quality, and prepare written feedback for emotion classification.")
    st.info("Analyze usable feedback with a pretrained English emotion model. First use downloads the model; inference runs locally on CPU.")
    if 'intake' not in st.session_state:
        st.session_state['intake'] = {'source': 'Sample dataset', 'identity': None,
                                      'uploaded_dataset': None, 'uploaded_identity': None}
    state = st.session_state['intake']
    previous_source = state['source']
    st.session_state.setdefault('_intake_source', previous_source)
    source = st.radio("Data source", ('Sample dataset', 'Upload CSV'), horizontal=True, key='_intake_source')
    if source != previous_source:
        reset_dataset(state, source, None, None)
    if source == 'Sample dataset':
        st.caption("48 synthetic student responses · No real student information or emotion labels.")
        if state['identity'] != 'sample':
            try:
                reset_dataset(state, source, 'sample', load_sample_data())
            except DatasetValidationError as exc:
                st.error(str(exc))
                return
    else:
        uploaded = st.file_uploader("Upload a UTF-8 CSV", type=['csv'], key='_intake_upload',
                                    help=f"Up to {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB, {MAX_ROWS:,} rows, and {MAX_COLUMNS} columns.")
        st.caption("Use anonymized feedback. Files stay in session memory and are not saved by the application. The last valid upload remains available when navigating or switching sources.")
        if uploaded is not None:
            identity = hashlib.sha256(uploaded.getbuffer()).hexdigest() + ':' + uploaded.name
            if identity != state.get('uploaded_identity'):
                try:
                    loaded = load_csv(uploaded, uploaded.name)
                except DatasetValidationError as exc:
                    state['uploaded_dataset'] = None
                    state['uploaded_identity'] = None
                    reset_dataset(state, source, None, None)
                    st.error(str(exc))
                else:
                    state['uploaded_dataset'] = loaded
                    state['uploaded_identity'] = identity
                    reset_dataset(state, source, identity, loaded)
        if state.get('active_dataset') is None and state.get('uploaded_dataset') is not None:
            reset_dataset(state, source, state['uploaded_identity'], state['uploaded_dataset'])
    data = state.get('active_dataset')
    if data is None:
        st.button("Analyze Feedback", disabled=True, type='primary')
        return

    options = [None, *data.columns]
    feedback = _selection('Feedback text column', options, state['feedback_column'], 'feedback_column')
    with st.expander('Optional context columns'):
        st.caption('Map fields for future analytics. ISO dates/timestamps are supported; ratings have no assumed scale.')
        for role in CONTEXT_COLUMNS:
            _selection(role.replace('_', ' ').title(), options, state['contexts'][role], 'contexts', role)
    contexts = state['contexts']
    preparation_key = (state['identity'], feedback, tuple(contexts.items()))
    if state.get('preparation_key') != preparation_key:
        invalidate_analysis(state)
        state['prepared'] = None
        state['profile'] = profile_dataset(data, contexts)
        state['preparation_key'] = preparation_key
        if feedback is not None:
            try:
                state['prepared'] = prepare_feedback(data, feedback, contexts)
            except DatasetValidationError as exc:
                st.error(str(exc))
    result = state['prepared']
    if result is None:
        st.warning('Select the column containing written student feedback to prepare the dataset.')
    else:
        st.markdown('#### Preparation summary')
        labels = [('Total Rows', 'total'), ('Usable Feedback', 'valid'), ('Missing Feedback', 'missing'),
                  ('Empty Feedback', 'empty'), ('Too Short', 'too_short'), ('Invalid / Non-text', 'non_text')]
        for start in (0, 3):
            for column, (label, status) in zip(st.columns(3), labels[start:start + 3]):
                column.metric(label, result.quality[status])
        st.caption(f'Usable feedback requires at least {MIN_FEEDBACK_LENGTH} characters after whitespace normalization. Original rows are retained, including unusable feedback.')
        if result.quality['valid'] < len(data):
            st.warning('Some feedback is unusable. Review the status column in Invalid / Unusable Rows; choose another text column or correct your source file as needed.')
        for role, counts in result.context_quality.items():
            st.caption(f"{role.replace('_', ' ').title()}: {counts['valid']:,} valid · {counts['missing']:,} missing/blank · {counts['invalid']:,} invalid")
            if counts['invalid']:
                st.warning(f"{counts['invalid']:,} {role.replace('_', ' ')} values could not be converted. Originals are retained; parsed values are empty. " +
                           ('Use ISO dates such as 2026-01-12.' if role == 'feedback_date' else 'Use finite numeric ratings. No rating scale is assumed.'))

    st.markdown('#### Dataset preview')
    st.caption(f'{len(data):,} rows · {len(data.columns)} columns · Each preview shows up to {PREVIEW_ROWS} rows')
    original_tab, prepared_tab, invalid_tab = st.tabs(['Original Data', 'Prepared Feedback', 'Invalid / Unusable Rows'])
    with original_tab:
        st.dataframe(data.head(PREVIEW_ROWS), hide_index=True, width='stretch')
    if result is not None:
        columns = list(dict.fromkeys([feedback, *(column for column in contexts.values() if column), *result.fields.values()]))
        with prepared_tab:
            st.dataframe(result.data[columns].head(PREVIEW_ROWS), width='stretch')
            st.caption('The displayed index matches the original row index. Preparation fields receive a numbered suffix if the input already uses that name.')
        with invalid_tab:
            unusable = result.data.loc[~result.data[result.fields['feedback_is_usable']], columns]
            st.caption('missing: null · empty: blank or whitespace-only · non_text: a non-string value · too_short: below the minimum length')
            if unusable.empty:
                st.success('All feedback rows meet the preparation rules.')
            else:
                st.dataframe(unusable.head(PREVIEW_ROWS), width='stretch')
    else:
        with prepared_tab:
            st.info('Select a feedback text column to see prepared rows.')
        with invalid_tab:
            st.info('Select a feedback text column to inspect quality issues.')

    profile = state['profile']
    with st.expander('Dataset profile'):
        st.write(f'{profile.row_count:,} rows · {profile.column_count} columns · {profile.duplicate_rows:,} duplicate rows beyond their first occurrence')
        st.dataframe(profile.columns, hide_index=True, width='stretch')
        for role, count in profile.context_unique.items():
            st.caption(f'Unique {role} values: {count:,}')
        st.caption('Missing values count nulls. Blank text is reported separately. Uploaded CSV cells retain their original text; no automatic numeric or null-token inference is applied.')
    if st.button('Analyze Feedback', type='primary', disabled=result is None or result.quality['valid'] == 0):
        invalidate_analysis(state)
        state['analysis_status'] = 'running'
        progress = st.progress(0.0, text='Loading model and analyzing feedback…')
        try:
            with st.spinner('Loading the model and running CPU inference…'):
                model = get_model()
                completed = classify_feedback(result, model, progress=progress.progress)
            state['analysis'] = completed
            state['analysis_status'] = 'complete'
        except EmotionError as exc:
            state['analysis_status'] = 'failed'
            state['analysis_error'] = str(exc)
        finally:
            progress.empty()
    if state.get('analysis_status') == 'failed':
        st.error(state['analysis_error'])
        st.info('No predictions were saved. Correct the issue and click Analyze Feedback to retry.')
    elif state.get('analysis_status') == 'complete':
        st.success('Emotion classification complete. Results remain available when navigating.')
    render_results(state)
