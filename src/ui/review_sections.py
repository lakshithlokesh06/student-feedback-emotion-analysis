"""Annotation widgets and descriptive review summaries."""
import pandas as pd
import plotly.express as px
import streamlit as st
from src.config import MODEL_EMOTIONS, REVIEW_STATUSES, REVIEW_NOTE_MAX_LENGTH, REVIEW_PRIORITY_MODES, HIGH_CONFIDENCE_THRESHOLD
from src.review.queue import QueueFilters, filter_queue, prioritize
from src.review.state import save_review, annotated_queue, clear_progress, move_item
from src.review.validation import ReviewValidationError
from src.review.metrics import progress, agreement, correction_bands, annotation_comparison, observations, small_sample_warnings
from src.review.export import reviewed_export, curated_export, audit_export


def current_workspace():
    return st.session_state['human_review']['workspaces'][st.session_state['human_review']['source']]


def _save_preference(name, is_filter=True):
    workspace = current_workspace()
    if is_filter:
        setattr(workspace['filters'], name, st.session_state['_review_' + name])
    else:
        workspace[name] = st.session_state['_review_' + name]


def _reset_filters():
    workspace = current_workspace()
    workspace['filters'] = QueueFilters()
    workspace['priority'] = 'All predictions'


def render_filters(data, workspace):
    modes = list(REVIEW_PRIORITY_MODES)
    if 'ambiguous_prediction' not in data:
        modes.remove('Ambiguous predictions')
    if 'is_correct' not in data:
        modes.remove('Misclassified evaluation rows')
    st.session_state['_review_priority'] = workspace['priority']
    st.selectbox('Prioritize review', modes, key='_review_priority', on_change=_save_preference, args=('priority', False))
    if workspace['priority'] == 'Specific emotion':
        st.session_state['_review_priority_emotion'] = workspace['priority_emotion']
        st.selectbox('Priority emotion', MODEL_EMOTIONS, key='_review_priority_emotion', on_change=_save_preference, args=('priority_emotion', False))
    st.caption(f'High-confidence review uses scores ≥ {HIGH_CONFIDENCE_THRESHOLD:.0%}. Attention review selects anger, fear, sadness, and disgust; it does not rank students or infer risk.')
    with st.expander('Review queue filters'):
        for name, label, options in [('statuses','Review status',REVIEW_STATUSES), ('predicted','Predicted emotion',MODEL_EMOTIONS), ('reviewed','Reviewed emotion',MODEL_EMOTIONS)]:
            st.session_state['_review_' + name] = getattr(workspace['filters'], name)
            st.multiselect(label, options, key='_review_' + name, on_change=_save_preference, args=(name,))
        for column, name in [('course','courses'),('subject','subjects'),('semester','semesters')]:
            if column in data and data[column].notna().any():
                choices = sorted(data[column].dropna().astype('string').unique())
                st.session_state['_review_' + name] = getattr(workspace['filters'], name)
                st.multiselect(column.title(), choices, key='_review_' + name, on_change=_save_preference, args=(name,))
        st.session_state['_review_confidence'] = workspace['filters'].confidence
        st.slider('Confidence range', 0.0, 1.0, key='_review_confidence', on_change=_save_preference, args=('confidence',))
        for field, name, label in [('low_confidence','low_confidence','Low-confidence flag'), ('ambiguous_prediction','ambiguity','Ambiguity flag')]:
            if field in data:
                st.session_state['_review_' + name] = getattr(workspace['filters'], name)
                st.selectbox(label, ['All','Yes','No'], key='_review_' + name, on_change=_save_preference, args=(name,))
        st.button('Reset Filters', on_click=_reset_filters)
    return prioritize(filter_queue(data, workspace['filters']), workspace['priority'], workspace['priority_emotion'])


def render_progress(data):
    stats = progress(data)
    for start in (0, 4):
        cards = [('Total Items',stats['total']),('Reviewed',stats['reviewed']),('Accepted',stats['accepted']),('Corrected',stats['corrected']),
                 ('Uncertain',stats['uncertain']),('Skipped',stats['skipped']),('Remaining',stats['remaining'])][start:start+4]
        for column, (label, value) in zip(st.columns(len(cards)), cards):
            column.metric(label, value)
    st.progress(stats['completion'], text=f"{stats['completion']:.0%} reviewed")
    st.caption('Progress uses the full source queue. Reviewed = accepted + corrected + uncertain. Remaining = unreviewed + skipped; skipped items are deferred. Agreement uses accepted/corrected decisions only.')
    with st.expander('Review status definitions'):
        st.write('Unreviewed: no saved decision. Accepted: reviewer agrees with the prediction. Corrected: reviewer selected a different supported emotion. Uncertain: no confident single-label decision. Skipped: review intentionally deferred.')


def _select_item(ids):
    position = st.session_state['_review_jump']
    current_workspace()['current_id'] = ids[position-1] if position else None


def _move(ids, offset):
    workspace = current_workspace()
    workspace['current_id'] = move_item(ids, workspace['current_id'], offset)


def _first(ids):
    workspace = current_workspace()
    workspace['current_id'] = next((item for item in ids if item not in workspace['records']), None)


def _clear_selection():
    current_workspace()['current_id'] = None


def _clear_progress():
    clear_progress(current_workspace())


def _save(reviewer, action_key, label_key, note_key, ids, advance):
    workspace = current_workspace()
    current = workspace['current_id']
    action = st.session_state[action_key]
    status = {'Accept prediction':'accepted','Choose another emotion':'corrected','Mark uncertain':'uncertain','Skip for now':'skipped'}[action]
    label = workspace['queue'].loc[current, 'predicted_emotion'] if status == 'accepted' else st.session_state.get(label_key)
    try:
        save_review(workspace, current, status, label, st.session_state[note_key], reviewer)
        workspace['message'] = 'Decision saved. Original prediction remains unchanged.'
        workspace['error'] = None
        if advance:
            workspace['current_id'] = move_item(ids, current, 1)
    except ReviewValidationError as exc:
        workspace['error'] = str(exc)
        workspace['message'] = None


def render_card(filtered, workspace, reviewer):
    ids = filtered.index.tolist()
    if workspace['current_id'] is not None and workspace['current_id'] not in ids:
        workspace['current_id'] = ids[0] if ids else None
    if not ids:
        st.info('No items match this queue. Reset Filters or choose another priority mode; saved annotations are retained.')
        return
    st.caption(f'{len(ids)} items in the current queue. Save edits before navigating; only saved decisions persist.')
    st.session_state['_review_jump'] = ids.index(workspace['current_id']) + 1 if workspace['current_id'] in ids else 0
    st.number_input('Jump to item (0 clears selection)', min_value=0, max_value=len(ids), step=1, key='_review_jump', on_change=_select_item, args=(ids,))
    left, middle, right = st.columns(3)
    left.button('Previous', on_click=_move, args=(ids,-1), disabled=not workspace['current_id'] or workspace['current_id'] == ids[0])
    middle.button('Next', on_click=_move, args=(ids,1), disabled=not workspace['current_id'] or workspace['current_id'] == ids[-1])
    right.button('First unreviewed item', on_click=_first, args=(ids,))
    st.button('Clear current item selection', on_click=_clear_selection)
    current = workspace['current_id']
    if current is None:
        st.info('Select a queue item to review. If all matching items have a saved status, none is unreviewed.')
        return
    row = filtered.loc[current]
    saved = workspace['records'].get(current, {})
    with st.container(border=True):
        st.markdown('#### Feedback under review')
        st.caption('Review ID: ' + current[:12])
        st.text(row.feedback)
        st.write(f'Prediction: {row.predicted_emotion} · Confidence: {row.emotion_confidence:.1%} · Status: {row.review_status}')
        for column in ('course','subject','semester','rating','feedback_date'):
            if column in row and pd.notna(row[column]):
                st.text(f'{column.replace("_"," ").title()}: {row[column]}')
        if 'true_emotion' in row:
            st.write(f'Evaluation label: {row.true_emotion} · Model matches reference: {bool(row.is_correct)}')
            st.caption('This reference is another annotation, not an infallible answer. Your review is stored separately.')
        if 'second_emotion' in row:
            st.write(f'Runner-up: {row.second_emotion} ({row.second_emotion_confidence:.1%}) · Ambiguous: {bool(row.ambiguous_prediction)} · Truncated: {bool(row.was_truncated)}')
        else:
            st.caption('Runner-up, ambiguity, and per-row truncation details are unavailable for normal analysis; review does not trigger extra inference.')
        st.caption('Single-label task: select a dominant emotion when reasonable, or mark uncertain for mixed emotions or insufficient context.')
        token = f'{current}_{workspace["revision"]}'
        action_key, label_key, note_key = [f'_review_{name}_{token}' for name in ('action','label','note')]
        actions = ['Accept prediction','Choose another emotion','Mark uncertain','Skip for now']
        defaults = {'accepted':0,'corrected':1,'uncertain':2,'skipped':3}
        action = st.radio('Review decision', actions, index=defaults.get(saved.get('review_status'),0), key=action_key)
        if action == 'Choose another emotion':
            options = [None, *(emotion for emotion in MODEL_EMOTIONS if emotion != row.predicted_emotion)]
            previous = saved.get('reviewed_emotion')
            st.selectbox('Reviewed Emotion', options, index=options.index(previous) if previous in options else 0, format_func=lambda value: 'Choose an emotion' if value is None else value, key=label_key)
        elif action == 'Accept prediction':
            st.caption(f'Reviewed Emotion will be {row.predicted_emotion}.')
        else:
            st.caption('Reviewed Emotion will remain empty; no emotion selection is required.')
        st.text_area('Optional review note', value=saved.get('review_note',''), max_chars=REVIEW_NOTE_MAX_LENGTH, key=note_key)
        left, right = st.columns(2)
        left.button('Save Review', on_click=_save, args=(reviewer, action_key, label_key, note_key, ids, False), type='primary')
        right.button('Save & Next', on_click=_save, args=(reviewer, action_key, label_key, note_key, ids, True))
        if saved:
            st.caption(f"Last saved by {saved['reviewer_id']} · {saved['reviewed_at']}")


def render_summary(data):
    st.subheader('Model and human review')
    stats, matrix, transitions = agreement(data)
    if stats['agreement'] is None:
        st.info('Accept or correct a prediction to see agreement statistics. Uncertain and skipped reviews are reported separately.')
    else:
        st.metric('Model-human agreement', f"{stats['agreement']:.1%}")
        st.caption('Agreement with this session’s annotations, not objective accuracy. Different reviewers can disagree; these observations describe only the reviewed subset.')
        means = pd.DataFrame({'Decision':['Accepted','Corrected'], 'Count':[stats['accepted'],stats['corrected']], 'Average model confidence':[stats['accepted_confidence'],stats['corrected_confidence']]})
        st.dataframe(means, hide_index=True, column_config={'Average model confidence':st.column_config.NumberColumn(format='%.3f')})
        if stats['accepted'] + stats['corrected'] >= 3:
            figure = px.imshow(matrix, text_auto=True, color_continuous_scale='Blues', labels={'x':'Human reviewed label','y':'Model prediction','color':'Count'}, aspect='auto')
            st.plotly_chart(figure, width='stretch', config={'modeBarButtonsToRemove':['toImage']})
        else:
            st.caption('The agreement matrix appears after at least three accepted/corrected decisions.')
        st.markdown('#### Correction patterns')
        st.dataframe(transitions, hide_index=True, width='stretch')
        left, right = st.columns(2)
        left.dataframe(transitions.groupby('predicted_emotion')['Count'].sum().reset_index(), hide_index=True)
        right.dataframe(transitions.groupby('reviewed_emotion')['Count'].sum().reset_index(), hide_index=True)
        st.dataframe(correction_bands(data), hide_index=True, column_config={'Correction_rate':st.column_config.NumberColumn(format='%.3f')})
        st.caption('Correction rates use accepted/corrected reviews within each confidence band; empty bands have no rate estimate.')
        if 'true_emotion' in data:
            st.markdown('#### Model / evaluation label / human annotation')
            st.dataframe(annotation_comparison(data), hide_index=True, width='stretch')
            st.caption('These categories compare annotations; no source is assumed infallible.')
    for warning in small_sample_warnings(data):
        st.warning(warning)
    for line in observations(data):
        st.write(line)


def render_exports(workspace):
    st.subheader('Export annotations')
    left, middle, right = st.columns(3)
    left.download_button('Reviewed Feedback CSV', reviewed_export(workspace), 'reviewed_feedback.csv', 'text/csv')
    middle.download_button('Curated Labeled Dataset CSV', curated_export(workspace), 'curated_labeled_feedback.csv', 'text/csv')
    right.download_button('Annotation Audit CSV', audit_export(workspace), 'annotation_audit.csv', 'text/csv')
    st.caption('Exports use the full source queue, not active filters. Reviewed export includes all saved statuses; curated data includes only accepted/corrected labels. Audit contains every save event, including revisions. Export before ending the session. Import/restore is planned, not implemented.')
    with st.expander('Clear review progress'):
        st.warning('Clearing removes all saved annotations and audit events for this source in this session. Original predictions and the other review source remain intact. Export first if you need a copy.')
        st.button('Clear ALL saved review progress for this source', on_click=_clear_progress)
