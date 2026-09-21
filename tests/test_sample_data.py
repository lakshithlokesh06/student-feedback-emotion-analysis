import pandas as pd
import pytest

from src.config import REQUIRED_DATASET_COLUMNS, SAMPLE_DATASET_PATH
from src.data.sample_data import DatasetValidationError, load_sample_data, validate_sample_data


def test_sample_dataset_contract():
    data = load_sample_data()
    assert set(REQUIRED_DATASET_COLUMNS).issubset(data.columns)
    assert not data.empty
    assert 40 <= len(data) <= 50
    assert data.feedback_id.is_unique
    assert data.feedback.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    assert not any('emotion' in column.lower() for column in data.columns)
    assert data.rating.between(1, 5).all()
    assert pd.to_datetime(data.feedback_date, errors='coerce').notna().all()


def test_sample_path_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert SAMPLE_DATASET_PATH.is_absolute()
    assert not load_sample_data().empty


def test_missing_file(tmp_path):
    with pytest.raises(DatasetValidationError, match='not found'):
        load_sample_data(tmp_path / 'missing.csv')


@pytest.mark.parametrize('content', ['', 'a,b\n1,2,3\n4,5,6,7\n', '\xff'])
def test_unreadable_csv(tmp_path, content):
    path = tmp_path / 'broken.csv'
    path.write_bytes(content.encode('latin-1'))
    with pytest.raises(DatasetValidationError, match='Could not read'):
        load_sample_data(path)


def test_missing_columns():
    with pytest.raises(DatasetValidationError, match='missing required columns: feedback'):
        validate_sample_data(load_sample_data().drop(columns=['feedback']))


def test_empty_dataset():
    with pytest.raises(DatasetValidationError, match='empty'):
        validate_sample_data(pd.DataFrame(columns=REQUIRED_DATASET_COLUMNS))


def test_duplicate_ids():
    data = load_sample_data()
    data.loc[1, 'feedback_id'] = data.loc[0, 'feedback_id']
    with pytest.raises(DatasetValidationError, match='duplicated'):
        validate_sample_data(data)


@pytest.mark.parametrize('value', [None, '', '   '])
def test_blank_ids(value):
    data = load_sample_data()
    data.loc[0, 'feedback_id'] = value
    with pytest.raises(DatasetValidationError, match='must not be blank'):
        validate_sample_data(data)


@pytest.mark.parametrize('value', [None, '', '   ', 42])
def test_unusable_feedback(value):
    data = load_sample_data()
    data['feedback'] = data['feedback'].astype(object)
    data.loc[0, 'feedback'] = value
    with pytest.raises(DatasetValidationError, match='non-empty text'):
        validate_sample_data(data)
