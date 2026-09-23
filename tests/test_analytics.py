from io import BytesIO
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.analytics.metrics import metrics, analytics_view, filtered_export
from src.analytics.distributions import emotion_distribution
from src.analytics.filters import DashboardFilters, apply_filters
from src.analytics.comparisons import group_comparison, rating_comparison
from src.analytics.trends import temporal_distribution, default_frequency
from src.analytics.insights import attention_feedback, descriptive_insights
from src.emotion.classifier import AnalysisResult


@pytest.fixture
def data():
    return pd.DataFrame({'emotion_label': ['joy', 'fear', 'joy', 'anger', None],
                         'emotion_confidence': [.9, .4, .7, .5, float('nan')],
                         'emotion_status': ['analyzed'] * 4 + ['not_analyzed'],
                         'course': ['A', 'B', 'A', 'B', None],
                         'subject': ['X', 'X', 'Y', 'Y', None],
                         'semester': ['Fall', 'Fall', 'Spring', 'Spring', None],
                         'rating': [5, 2, 4, float('nan'), 1],
                         'date': ['2026-01-01', '2026-01-04', '2026-02-02', 'bad', None]})


def test_metrics_distribution(data):
    stats = metrics(data, .5)
    assert stats['total'] == 5 and stats['analyzed'] == 4
    assert stats['dominant'] == 'joy'
    assert stats['average'] == pytest.approx(.625)
    assert stats['median'] == pytest.approx(.6)
    assert stats['low'] == 1 and stats['low_percentage'] == 25
    dist = emotion_distribution(data).set_index('Emotion')
    assert dist.loc['joy', 'Count'] == 2
    assert dist.loc['joy', 'Percentage'] == 50
    assert dist.Percentage.sum() == 100


@pytest.mark.parametrize('column', ['course', 'subject', 'semester'])
def test_groups(data, column):
    summary, composition, count = group_comparison(data, column)
    assert count == 2
    assert summary.Count.sum() == 4
    assert set(summary.Count) == {2}
    assert composition.groupby('Group').Percentage.sum().eq(100).all()
    assert summary['Average confidence'].notna().all()
    assert len(group_comparison(data, column, limit=1)[0]) == 1


def test_rating_aggregation(data):
    averages, distribution, excluded = rating_comparison(data)
    assert excluded == 1
    assert averages.set_index('emotion_label').loc['joy', 'mean'] == 4.5
    assert distribution.Count.sum() == 3
    data.loc[0, 'rating'] = float('inf')
    assert rating_comparison(data)[2] == 2


@pytest.mark.parametrize('frequency,periods', [('Day', 33), ('Week', 6), ('Month', 2)])
def test_temporal(data, frequency, periods):
    dist, excluded = temporal_distribution(data, frequency)
    assert excluded == 1
    assert dist.Count.sum() == 3
    assert dist.Period.nunique() == periods
    assert dist.loc[dist.Volume.gt(0)].groupby('Period').Percentage.sum().eq(100).all()
    assert dist.loc[dist.Volume.eq(0), 'Percentage'].isna().all()


def test_date_defaults_and_invalid(data):
    assert default_frequency(data) == 'Week'
    assert default_frequency(data.iloc[:2]) == 'Day'
    data['date'] = 'invalid'
    assert temporal_distribution(data, 'Day')[0].empty
    assert temporal_distribution(data, 'Day')[1] == 4


@pytest.mark.parametrize('filters,count', [
    (DashboardFilters(), 5), (DashboardFilters(emotions=['joy']), 2),
    (DashboardFilters(courses=['A']), 2), (DashboardFilters(subjects=['X']), 2),
    (DashboardFilters(semesters=['Spring']), 2), (DashboardFilters(ratings=[5]), 1),
    (DashboardFilters(confidence='Low confidence'), 1),
    (DashboardFilters(confidence='At or above threshold'), 3),
    (DashboardFilters(emotions=['joy'], courses=['A'], subjects=['Y']), 1),
    (DashboardFilters(emotions=['sadness']), 0),
])
def test_filters(data, filters, count):
    before = data.copy(deep=True)
    assert len(apply_filters(data, filters, .5)) == count
    assert_frame_equal(data, before)


def test_attention_insights(data):
    assert attention_feedback(data).emotion_label.tolist() == ['anger', 'fear']
    insights = descriptive_insights(data, .5)
    assert insights == descriptive_insights(data, .5)
    assert '50.0%' in insights[0] and '25.0%' in insights[1]
    assert 'validated accuracy' in insights[2]


def test_empty_one_and_missing_context(data):
    empty = data.iloc[:0]
    assert metrics(empty, .5)['average'] is None
    assert emotion_distribution(empty).Count.sum() == 0
    assert 'No analyzed' in descriptive_insights(empty, .5)[0]
    assert attention_feedback(empty).empty
    one = data.iloc[:1]
    assert metrics(one, .5)['median'] == .9
    assert emotion_distribution(one).Percentage.max() == 100
    bare = data.drop(columns=['course', 'subject', 'semester', 'rating', 'date'])
    for column in ['course', 'subject', 'semester']:
        assert group_comparison(bare, column)[0].empty
    assert rating_comparison(bare)[0].empty
    assert temporal_distribution(bare, 'Month')[0].empty


def test_view_mapping_and_export_with_duplicate_source_index(data):
    original = data.rename(columns={'emotion_label': 'emotion_label_2'})
    original.index = [5] * 5
    fields = {name: name for name in ('emotion_confidence', 'emotion_status')}
    fields['emotion_label'] = 'emotion_label_2'
    result = AnalysisResult(original, fields, {}, .5)
    before = original.copy(deep=True)
    view = analytics_view(result, {}, {'course': 'course'})
    selected = apply_filters(view, DashboardFilters(emotions=['joy']), .5)
    exported = pd.read_csv(BytesIO(filtered_export(result, selected)))
    assert len(exported) == 2
    assert list(exported.columns) == list(original.columns)
    assert exported.emotion_label_2.eq('joy').all()
    assert_frame_equal(original, before)
