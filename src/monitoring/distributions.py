import numpy as np
import pandas as pd


def distribution(reference, current, categories=None, name='category'):
    a, b = reference.dropna().value_counts(), current.dropna().value_counts()
    categories = list(categories) if categories is not None else sorted(set(a.index) | set(b.index), key=str)
    table = pd.DataFrame({name: categories})
    for prefix, counts in [('reference', a), ('current', b)]:
        table[prefix + '_count'] = [int(counts.get(c, 0)) for c in categories]
        table[prefix + '_percentage'] = table[prefix + '_count'] / counts.sum() * 100 if counts.sum() else np.nan
    table['percentage_point_change'] = table.current_percentage - table.reference_percentage
    return table


def jensen_shannon(p, q):
    """Base-2 Jensen–Shannon divergence, bounded [0, 1], not its square root."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.shape != q.shape or p.ndim != 1 or not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any() or p.sum() <= 0 or q.sum() <= 0:
        raise ValueError('Provide aligned, finite nonnegative distributions with positive totals.')
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    def kl(x):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / m[mask])))
    return max(0., min(1., (kl(p) + kl(q)) / 2))


def ks_distance(a, b):
    """Maximum empirical CDF difference [0,1]; effect size only, no p-value."""
    a, b = np.sort(a.dropna()), np.sort(b.dropna())
    if not len(a) or not len(b):
        return float('nan')
    points = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, points, side='right') / len(a) - np.searchsorted(b, points, side='right') / len(b))))
