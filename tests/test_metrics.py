import numpy as np

from musicdetect.eval import (
    apply_temperature,
    auc,
    ece,
    eer,
    fit_temperature,
    grouped_bootstrap,
    holm,
    paired_delta,
    rates_at_threshold,
    threshold_at_fpr,
)


def _toy(rng, n_groups=60, per=10, sep=1.0):
    g = np.repeat(np.arange(n_groups), per)
    y = np.tile([0, 1], n_groups * per // 2)
    return y, y * sep + rng.normal(0, 1, len(y)), g


def test_auc_eer_on_separable_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auc(y, s) == 1.0 and eer(y, s) == 0.0
    assert auc(np.zeros(5), np.arange(5)) != auc(np.zeros(5), np.arange(5))      # NaN : une seule classe


def test_threshold_at_fpr_and_rates():
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, 100000)
    thr = threshold_at_fpr(real, 0.01)
    assert abs(np.mean(real > thr) - 0.01) < 0.002
    y = np.r_[np.zeros(1000), np.ones(1000)]
    s = np.r_[rng.normal(0, 1, 1000), rng.normal(3, 1, 1000)]
    tpr, fpr = rates_at_threshold(y, s, thr)
    assert tpr > 0.5 and fpr < 0.05


def test_grouped_bootstrap_ci_contains_point_and_is_wider_than_naive(rng):
    y, s, g = _toy(rng)
    a, lo, hi = grouped_bootstrap(y, s, g, B=300, seed=1)
    assert lo < a < hi and 0.5 < lo


def test_grouped_bootstrap_widens_with_group_correlation():
    """Des segments d'un même groupe très corrélés : l'IC groupé doit être plus large que l'IC par segments."""
    rng = np.random.default_rng(0)
    n_g, per = 40, 20
    g = np.repeat(np.arange(n_g), per)
    y = np.repeat(rng.integers(0, 2, n_g), per)
    s = y + np.repeat(rng.normal(0, 1, n_g), per) + 0.05 * rng.normal(size=n_g * per)
    _, lo_g, hi_g = grouped_bootstrap(y, s, g, B=300, seed=1)
    _, lo_s, hi_s = grouped_bootstrap(y, s, np.arange(len(y)), B=300, seed=1)
    assert (hi_g - lo_g) > (hi_s - lo_s)


def test_paired_delta_detects_difference_and_not_equality(rng):
    y, sA, g = _toy(rng, sep=1.5)
    sB = y * 0.5 + rng.normal(0, 1, len(y))
    d = paired_delta(y, sA, sB, g, B=300)
    assert d["excludes_zero"] and d["delta"] > 0 and d["p"] < 0.05
    same = paired_delta(y, sA, sA, g, B=100)
    assert same["delta"] == 0 and not same["excludes_zero"]


def test_holm_matches_manual_computation():
    out = holm([0.01, 0.04, 0.03])
    assert [round(p, 6) for p, _ in out] == [0.03, 0.06, 0.06]
    assert [r for _, r in out] == [True, False, False]


def test_ece_and_temperature(rng):
    y = rng.integers(0, 2, 5000)
    z = (2 * y - 1) * 0.8 + rng.normal(0, 1, 5000)
    p_over = 1 / (1 + np.exp(-4 * z))              # sur-confiant
    T = fit_temperature(y, p_over)
    assert T > 2.0
    assert ece(y, apply_temperature(p_over, T)) < ece(y, p_over)
