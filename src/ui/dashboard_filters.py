"""Filter widget state is separate from analysis state and survives navigation."""
from dataclasses import asdict
import streamlit as st
from src.analytics.filters import DashboardFilters
from src.config import MODEL_EMOTIONS


def _save(name):
    st.session_state['dashboard_filters'][name] = st.session_state['_dash_' + name]


def reset_filters():
    st.session_state['dashboard_filters'] = asdict(DashboardFilters())
    for key in list(st.session_state):
        if key.startswith('_dash_'):
            del st.session_state[key]


def render_filters(data, generation, threshold):
    if st.session_state.get('dashboard_generation') != generation:
        reset_filters()
        st.session_state['dashboard_generation'] = generation
    settings = st.session_state['dashboard_filters']
    with st.expander('Dashboard filters'):
        st.caption('Empty selections include all values, including missing context. Selecting a value excludes missing values for that field. All dashboard sections use these filters.')
        for name, column in [('emotions', 'emotion_label'), ('courses', 'course'), ('subjects', 'subject'), ('semesters', 'semester'), ('ratings', 'rating')]:
            if column not in data or data[column].dropna().empty:
                continue
            options = [e for e in MODEL_EMOTIONS if e in data[column].values] if name == 'emotions' else sorted(data[column].dropna().unique())
            settings[name] = [value for value in settings[name] if value in options]
            st.session_state['_dash_' + name] = settings[name]
            st.multiselect(name.title(), options, key='_dash_' + name, on_change=_save, args=(name,))
        st.session_state['_dash_confidence'] = settings['confidence']
        st.selectbox('Confidence level', ['All', 'Low confidence', 'At or above threshold'], key='_dash_confidence', on_change=_save, args=('confidence',))
        st.caption(f'Low confidence means below {threshold:.0%}; confidence filters include analyzed rows only.')
        st.button('Reset Filters', on_click=reset_filters)
    return DashboardFilters(**settings)
