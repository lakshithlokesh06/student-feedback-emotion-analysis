from src.monitoring.validation import validate_monitoring


def new_state():
    return dict(reference=None, current=None, sources={}, results=None, filters={})


def replace_dataset(state, role, data, source, threshold):
    if role not in ('reference', 'current'):
        raise ValueError('Unknown monitoring role.')
    validated = validate_monitoring(data, threshold)
    state[role] = validated
    state['sources'][role] = source
    state['results'] = None
    state['filters'] = {}


def snapshot(state, data, threshold):
    replace_dataset(state, 'reference', data, 'Main analysis reference snapshot (session only)', threshold)
