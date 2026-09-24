"""Evaluation state transitions never touch the normal intake namespace."""

def replace_dataset(state, source, identity, dataset=None, error=None):
    state.update(source=source, identity=identity, dataset=dataset, results=None,
                 status='ready' if dataset is not None else 'empty', error=error,
                 filters={'true': [], 'predicted': [], 'correctness': 'All', 'confidence': 0.0})
