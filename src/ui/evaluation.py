import hashlib
import streamlit as st
from src.config import LABELED_SAMPLE_PATH, MODEL_EMOTIONS, PREVIEW_ROWS
from src.data.loader import load_csv
from src.data.validation import DatasetValidationError
from src.evaluation.validation import validate_evaluation
from src.evaluation.runner import run_evaluation
from src.evaluation.state import replace_dataset
from src.emotion.labels import EmotionError
from src.ui.model_cache import get_model
from src.ui import evaluation_sections as sections


def render_evaluation():
    st.subheader('Model Evaluation')
    st.write('Evaluation compares predictions with supplied reference labels. Confidence is a model score; accuracy measures agreement with labels. This pretrained model was not specifically trained on this application’s student-feedback dataset.')
    st.caption('Load labeled CSV → validate text and labels → run the existing model → compare metrics → inspect errors. This workspace is separate from normal feedback analysis.')
    if 'evaluation' not in st.session_state:
        state = {}
        replace_dataset(state, 'Synthetic labeled sample', None)
        st.session_state['evaluation'] = state
    state = st.session_state['evaluation']
    st.session_state.setdefault('_evaluation_source', state['source'])
    source = st.radio('Evaluation data source', ['Synthetic labeled sample','Upload labeled CSV'], key='_evaluation_source', horizontal=True)
    if source != state['source']:
        replace_dataset(state, source, None)
    if source == 'Synthetic labeled sample':
        st.warning('This bundled dataset is synthetic and intended only to demonstrate the evaluation workflow. It is not a benchmark for real-world model accuracy.')
        if state['identity'] != 'sample':
            try:
                with LABELED_SAMPLE_PATH.open('rb') as file:
                    dataset = validate_evaluation(load_csv(file, LABELED_SAMPLE_PATH.name))
                replace_dataset(state, source, 'sample', dataset)
            except (DatasetValidationError, OSError) as exc:
                state['error'] = str(exc) if isinstance(exc, DatasetValidationError) else 'The labeled sample could not be read. Restore data/sample_labeled_feedback.csv.'
    else:
        st.caption('Required columns: feedback, true_emotion. Optional context is preserved. UTF-8 CSV; existing 10 MiB, 100,000-row, and 100-column limits apply. No uploads are saved to disk.')
        upload = st.file_uploader('Upload labeled evaluation CSV', type=['csv'], key='_evaluation_upload')
        if upload is not None:
            identity = hashlib.sha256(upload.getbuffer()).hexdigest() + upload.name
            if identity != state['identity']:
                try:
                    dataset = validate_evaluation(load_csv(upload, upload.name))
                    replace_dataset(state, source, identity, dataset)
                except DatasetValidationError as exc:
                    replace_dataset(state, source, identity, error=str(exc))
        st.caption('The current evaluation upload remains in this session across navigation. Switching sources clears evaluation results and selections.')
    if state['error']:
        st.error(state['error'])
    dataset = state['dataset']
    if dataset is None:
        st.info('Choose a labeled sample or upload a labeled CSV to begin.')
        return
    frame = dataset.prepared.data
    st.write(f'{len(frame)} input rows · {dataset.valid_count} valid evaluation rows · {len(frame) - dataset.valid_count} rejected · {dataset.duplicate_count} duplicate normalized text/label pairs retained')
    st.caption('Accepted labels: ' + ', '.join(MODEL_EMOTIONS) + '. Labels are trimmed and lowercased; unrelated labels are never mapped. Original labels and rows are preserved.')
    st.dataframe(frame.head(PREVIEW_ROWS), hide_index=True, width='stretch')
    invalid = frame.loc[frame[dataset.fields['evaluation_validation']].ne('valid')]
    if len(invalid):
        st.warning('Rejected rows have missing/unsupported labels or unusable text. Correct the source labels/text if these rows should be evaluated; only valid rows contribute to metrics.')
        with st.expander('Rejected evaluation rows and reasons'):
            st.dataframe(invalid.head(PREVIEW_ROWS), hide_index=True, width='stretch')
    if st.button('Run Evaluation', type='primary', disabled=dataset.valid_count == 0):
        state['review_generation'] = state.get('review_generation', 0) + 1
        state.update(results=None, status='running', error=None)
        progress = st.progress(0.0)
        try:
            with st.spinner('Evaluating with the cached emotion model on CPU…'):
                result = run_evaluation(dataset, get_model(), progress=progress.progress)
            state.update(results=result, status='complete')
        except EmotionError as exc:
            state.update(status='failed', error=str(exc))
            st.error(str(exc))
        finally:
            progress.empty()
    result = state['results']
    if result is None:
        return
    st.success('Evaluation complete. Metrics describe this labeled dataset only.')
    view, stats, classes = sections.summary(result, source == 'Synthetic labeled sample')
    section = st.radio('Evaluation section', ['Performance','Confidence & Calibration','Error Analysis','Score Transparency'], horizontal=True)
    if section == 'Performance':
        sections.performance_section(view, stats, classes)
    elif section == 'Confidence & Calibration':
        sections.confidence_section(view)
    elif section == 'Error Analysis':
        sections.errors_section(result, view, state)
    else:
        sections.transparency_section(result, view)
