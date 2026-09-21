"""Session-state transitions, separate from widget rendering."""
from collections.abc import MutableMapping

import pandas as pd

from src.config import CONTEXT_COLUMNS


def reset_dataset(state: MutableMapping, source: str, identity: str | None,
                  data: pd.DataFrame | None) -> None:
    """Replace active input and invalidate every dependent selection/result."""
    state['source'] = source
    state['identity'] = identity
    state['active_dataset'] = data
    state['prepared'] = None
    state['preparation_key'] = None
    state['profile'] = None
    columns = [] if data is None else list(data.columns)
    state['feedback_column'] = 'feedback' if 'feedback' in columns else None
    state['contexts'] = {role: role if role in columns else None for role in CONTEXT_COLUMNS}
