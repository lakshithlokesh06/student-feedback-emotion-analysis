from io import BytesIO

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.loader import load_csv
from src.data.preparation import clean_feedback, prepare_feedback
from src.data.profiling import profile_dataset
from src.data.validation import DatasetValidationError, IntakeLimits
from src.ui.intake_state import reset_dataset


def read(text, name='feedback.csv', **limits):
    raw = text.encode('utf-8') if isinstance(text, str) else text
    return load_csv(BytesIO(raw), name, IntakeLimits(**limits))


def test_valid_csv_preserves_values_and_bom():
    data = read('\ufefffeedback,score\n"Great, thanks!",004\n"Line one\nLine two",NA\n', 'FEEDBACK.CSV')
    assert list(data.columns) == ['feedback', 'score']
    assert data.iloc[0].tolist() == ['Great, thanks!', '004']
    assert data.iloc[1].tolist() == ['Line one\nLine two', 'NA']


@pytest.mark.parametrize('content,name,match', [
    ('x\na', 'file.txt', '.csv'), ('', 'a.csv', 'empty'),
    ('x\n', 'a.csv', 'empty'), ('   ', 'a.csv', 'blank'),
    ('\n', 'a.csv', 'at least one column'),
    ('a,\n1,2', 'a.csv', 'blank'), ('a, \n1,2', 'a.csv', 'blank'),
    ('a,a\n1,2', 'a.csv', 'Duplicate'), ('a, a \n1,2', 'a.csv', 'Duplicate'),
    ('a,b\n1,2,3', 'a.csv', 'fields'), ('a,b\n1', 'a.csv', 'fields'),
    ('a\n"unfinished', 'a.csv', 'Malformed'),
    (b'a\n\xff', 'a.csv', 'UTF-8'), ('a\n\x00', 'a.csv', 'null bytes'),
])
def test_invalid_csv(content, name, match):
    with pytest.raises(DatasetValidationError, match=match):
        read(content, name)


@pytest.mark.parametrize('content,limits,match', [
    ('a\nabc', {'max_bytes': 4}, 'MB'),
    ('a\n1\n2', {'max_rows': 1}, 'rows'),
    ('a,b\n1,2', {'max_columns': 1}, 'columns'),
])
def test_limits(content, limits, match):
    with pytest.raises(DatasetValidationError, match=match):
        read(content, **limits)


def test_exact_limits_and_blank_rows():
    data = read('a\nx\n\n', max_bytes=6, max_rows=2, max_columns=1)
    assert data.a.tolist() == ['x', '']


@pytest.mark.parametrize('value,clean,status', [
    (None, None, 'missing'), (pd.NA, None, 'missing'), (float('nan'), None, 'missing'),
    ('', '', 'empty'), (' \t\n ', '', 'empty'),
    (123, None, 'non_text'), (True, None, 'non_text'), (['text'], None, 'non_text'),
    ('ok', 'ok', 'too_short'), ('Yes', 'Yes', 'valid'),
    ('  I\t am\nNOT happy!  ', 'I am NOT happy!', 'valid'),
    ('I’m happy 😊', 'I’m happy 😊', 'valid'),
])
def test_feedback_rules(value, clean, status):
    assert clean_feedback(value) == (clean, status)


def test_preparation_preserves_original_and_handles_collisions():
    data = pd.DataFrame({'feedback': ['  Good! ', None, ' ', 1, 'ok'], 'feedback_clean': ['original'] * 5})
    original = data.copy(deep=True)
    result = prepare_feedback(data, 'feedback')
    assert_frame_equal(data, original)
    assert_frame_equal(result.data[data.columns], original)
    assert result.fields['feedback_clean'] == 'feedback_clean_2'
    assert result.data.feedback_clean_2.fillna('<null>').tolist() == ['Good!', '<null>', '', '<null>', 'ok']
    assert result.quality == {'valid': 1, 'missing': 1, 'empty': 1, 'non_text': 1, 'too_short': 1, 'total': 5}
    assert result.data.feedback_is_usable.tolist() == [True, False, False, False, False]


@pytest.mark.parametrize('selection', [None, 'absent'])
def test_missing_selection(selection):
    with pytest.raises(DatasetValidationError, match='Select'):
        prepare_feedback(pd.DataFrame({'text': ['hello']}), selection)


def test_context_selection_validation():
    with pytest.raises(DatasetValidationError, match='rating'):
        prepare_feedback(pd.DataFrame({'text': ['hello']}), 'text', {'rating': 'absent'})


def test_minimum_length():
    assert prepare_feedback(pd.DataFrame({'text': ['hello']}), 'text', minimum_length=6).quality['too_short'] == 1
    with pytest.raises(DatasetValidationError):
        prepare_feedback(pd.DataFrame({'text': ['hello']}), 'text', minimum_length=0)


def test_dates_and_ratings():
    data = pd.DataFrame({
        'text': ['Good class'] * 7,
        'date': ['2026-01-12', '2026-01-13T10:00:00+05:30', '2026-02-30', '01/02/2026', None, '', 123],
        'score': ['4.5', '-1', 'unrated', None, ' ', 'inf', '100'],
    }, index=[10, 20, 30, 40, 50, 60, 70])
    original = data.copy(deep=True)
    result = prepare_feedback(data, 'text', {'feedback_date': 'date', 'rating': 'score'})
    assert result.context_quality['feedback_date'] == {'valid': 2, 'missing': 2, 'invalid': 3}
    assert result.context_quality['rating'] == {'valid': 3, 'missing': 2, 'invalid': 2}
    assert result.data.loc[20, 'feedback_date_parsed'] == pd.Timestamp('2026-01-13T04:30:00Z')
    assert pd.isna(result.data.loc[30, 'feedback_date_parsed'])
    assert result.data.loc[10, 'rating_numeric'] == 4.5
    assert result.data.loc[70, 'rating_numeric'] == 100
    assert_frame_equal(data, original)
    assert_frame_equal(result.data[data.columns], original)


def test_profile_counts():
    data = pd.DataFrame({'text': ['Great', 'Great', None, ' '], 'class': ['A', 'A', 'B', '']})
    result = profile_dataset(data, {'course': 'class', 'subject': None})
    assert (result.row_count, result.column_count, result.duplicate_rows) == (4, 2, 1)
    assert result.context_unique == {'course': 2}
    assert result.columns['Column'].tolist() == ['text', 'class']
    assert result.columns['Missing values'].tolist() == [1, 0]
    assert result.columns['Blank text'].tolist() == [1, 1]
    assert result.columns['Data type'].tolist() == [str(dtype) for dtype in data.dtypes]


def test_reset_clears_dependent_state():
    state = {'prepared': 'stale', 'profile': 'stale', 'contexts': {'rating': 'old'}}
    data = pd.DataFrame({'text': ['Hello']})
    reset_dataset(state, 'Upload CSV', 'new', data)
    assert state['active_dataset'] is data
    assert state['prepared'] is None and state['profile'] is None
    assert state['feedback_column'] is None
    assert all(value is None for value in state['contexts'].values())
