"""Coverage, width, Wilson CI, and the Kupiec test behave correctly."""
import numpy as np
import pytest

from src import calibration as C


def test_empirical_coverage_and_width():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lo = np.array([-1.0, 0.5, 5.0, 2.9])   # points 0,1,3 inside; 2 outside
    hi = np.array([1.0, 1.5, 6.0, 3.1])
    assert C.empirical_coverage(y, lo, hi) == pytest.approx(0.75)
    assert C.average_width(lo, hi) == pytest.approx(np.mean(hi - lo))


def test_coverage_ci_brackets_estimate_and_in_unit_interval():
    lo, hi = C.coverage_ci(1000, 950)
    assert lo < 0.95 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_coverage_ci_narrows_with_n():
    _, wide_hi = C.coverage_ci(100, 95)
    wide = wide_hi - C.coverage_ci(100, 95)[0]
    narrow_lo, narrow_hi = C.coverage_ci(10000, 9500)
    narrow = narrow_hi - narrow_lo
    assert narrow < wide


def test_kupiec_pvalue_high_at_nominal():
    # Exactly the expected number of failures -> cannot reject -> p near 1.
    assert C.kupiec_test(n=1000, failures=50, level=0.95) == pytest.approx(1.0, abs=1e-6)


def test_kupiec_pvalue_low_when_far_off():
    # 12% failures against a 5% expectation -> strongly rejected.
    assert C.kupiec_test(n=1000, failures=120, level=0.95) < 1e-6


def test_kupiec_handles_zero_failures():
    p = C.kupiec_test(n=500, failures=0, level=0.99)
    assert 0.0 <= p <= 1.0        # must not raise on log(0)


def test_coverage_table_shape_and_metrics():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 2000)
    models = {
        "A": lambda lv: (np.full_like(y, -2.0), np.full_like(y, 2.0)),
        "B": lambda lv: (np.full_like(y, -3.0), np.full_like(y, 3.0)),
    }
    table = C.coverage_table(y, models, [0.90, 0.95])
    assert list(table.index) == ["A", "B"]
    assert ("90%", "coverage") in table.columns
    assert ("95%", "kupiec_p") in table.columns
