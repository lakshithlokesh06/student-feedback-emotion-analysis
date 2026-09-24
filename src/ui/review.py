import streamlit as st
from src.config import REVIEW_SOURCES, DEFAULT_REVIEWER_ID, REVIEWER_ID_MAX_LENGTH, CONTEXT_COLUMNS
from src.review.queue import create_queue
from src.review.state import new_workspace, annotated_queue
from src.ui import review_sections as sections


def _save_reviewer():
    st.session_state['human_review']['reviewer'] = st.session_state['_reviewer_identity']


def render_review():
    st.subheader('Human Review')
    st.write('Inspect automated emotion predictions, confirm or correct labels, and build annotated data for evaluation or future model development. Saving reviews does not retrain the model.')
    if 'human_review' not in st.session_state:
        st.session_state['human_review'] = dict(source=REVIEW_SOURCES[0], reviewer=DEFAULT_REVIEWER_ID, workspaces={})
    state = st.session_state['human_review']
    st.session_state.setdefault('_human_review_source', state['source'])
    source = st.radio('Review source', REVIEW_SOURCES, horizontal=True, key='_human_review_source')
    state['source'] = source
    st.session_state['_reviewer_identity'] = state['reviewer']
    reviewer = st.text_input('Optional local reviewer ID', key='_reviewer_identity', max_chars=REVIEWER_ID_MAX_LENGTH, on_change=_save_reviewer, help='A local export label, not an account. A nickname or non-identifying ID is sufficient.')
    if source == REVIEW_SOURCES[0]:
        parent = st.session_state.get('intake', {})
        result = parent.get('analysis')
        generation = parent.get('analytics_generation', 0)
        if result is not None:
            frame, fields = result.data, result.fields
            feedback, contexts = parent['feedback_column'], parent.get('contexts', {})
            revision = result.metadata.get('revision','')
    else:
        parent = st.session_state.get('evaluation', {})
        result = parent.get('results')
        generation = parent.get('review_generation', 0)
        if result is not None:
            frame, fields = result.analysis.data, result.fields
            feedback, contexts = 'feedback', {column:column for column in CONTEXT_COLUMNS if column in frame}
            revision = result.analysis.metadata.get('revision','')
    workspaces = state['workspaces']
    if result is None:
        workspaces.pop(source, None)
        st.info('No completed predictions are available for this source. Run Analyze Feedback or Model Evaluation first. The two review sources are never combined.')
        return
    previous = workspaces.get(source)
    if previous is None or previous['generation'] != generation:
        queue, identity = create_queue(frame, fields, source, feedback, contexts, revision)
        workspace = new_workspace(queue, identity, generation)
        workspace['current_id'] = queue.index[0] if len(queue) else None
        workspaces[source] = workspace
        if previous is not None:
            st.info('The prediction snapshot changed. Review progress for the previous snapshot has been cleared; the other source is unaffected.')
    workspace = workspaces[source]
    data = annotated_queue(workspace)
    st.caption(f'Source: {source} · {len(data)} classified rows eligible for review · Unclassified/rejected rows are excluded.')
    sections.render_progress(data)
    if data.empty:
        st.info('This source contains no classified rows to review.')
        return
    filtered = sections.render_filters(data, workspace)
    section = st.radio('Review workspace', ['Review Items','Agreement & Patterns','Exports'], horizontal=True)
    if workspace.get('error'):
        st.error(workspace['error'])
    elif workspace.get('message'):
        st.success(workspace['message'])
    if section == 'Review Items':
        sections.render_card(filtered, workspace, reviewer)
    elif section == 'Agreement & Patterns':
        sections.render_summary(data)
    else:
        sections.render_exports(workspace)
