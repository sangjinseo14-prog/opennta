"""Unit tests for the TrackMate threshold *distribution models*.

These models (``CS210``, ``Poly2``, ``Gaussian``) are the pure-math core of
the Fiji threshold step: each defines an (untruncated) peak-quality density,
its upper-tail probability, and the truncation normalizer the MLE fitter
divides by. They are Fiji-independent — the module docstring even states they
exist "so they can be unit-tested in isolation against synthetic quality
samples" — yet were previously reachable only through the Fiji-bound
integration test, which skips in CI.

The tests below assert the mathematical contract every model must satisfy
(normalised density, monotone decreasing tail, truncation normalizer in
``(0, 1]`` and approaching 1), so a regression in any model surfaces without a
Fiji binary.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from opennta.tracking.trackmate.fitting_models import (
    CS210Model,
    GaussianModel,
    Poly2Model,
    get_model_class,
)

# Representative, valid parameter sets per model (near-Gaussian shapes).
MODEL_PARAMS = {
    "Gaussian": (GaussianModel, {"mu": 50.0, "sigma": 5.0}),
    "CS210": (CS210Model, {"mu": 50.0, "sigma": 5.0, "kappa": 0.999}),
    "Poly2": (Poly2Model, {"mu": 50.0, "sigma": 5.0, "a0": 1.0, "a1": 0.0, "a2": 0.3}),
}
MODEL_IDS = list(MODEL_PARAMS)


@pytest.fixture(params=MODEL_IDS)
def model_and_params(request):
    cls, params = MODEL_PARAMS[request.param]
    return cls(), params


def test_get_model_class_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_class("does_not_exist")


def test_pdf_is_nonnegative_and_integrates_to_one(model_and_params):
    model, params = model_and_params
    mu, sigma = params["mu"], params["sigma"]
    grid = np.linspace(mu - 12 * sigma, mu + 12 * sigma, 6000)
    dens = model.pdf_untruncated(grid, **params)

    assert np.all(np.isfinite(dens))
    assert np.all(dens >= 0.0)

    # scipy.integrate.trapezoid is stable across NumPy 1.26 and 2.x; np.trapezoid
    # does not exist on the project's supported NumPy 1.26 floor.
    mass = float(integrate.trapezoid(dens, grid))
    assert mass == pytest.approx(1.0, abs=0.02)


def test_tail_probability_is_monotone_decreasing_in_unit_interval(model_and_params):
    model, params = model_and_params
    mu, sigma = params["mu"], params["sigma"]
    us = np.linspace(mu - 2 * sigma, mu + 6 * sigma, 25)
    tails = np.array([model.tail_probability(float(u), **params) for u in us])

    assert np.all(tails >= -1e-9) and np.all(tails <= 1.0 + 1e-9)
    # Non-increasing as the threshold rises (allow tiny numerical slack).
    assert np.all(np.diff(tails) <= 1e-6)
    # Far into the tail the survival probability is essentially zero.
    assert model.tail_probability(mu + 10 * sigma, **params) == pytest.approx(0.0, abs=1e-3)


def test_truncation_normalizer_in_unit_interval_and_approaches_one(model_and_params):
    model, params = model_and_params
    mu, sigma = params["mu"], params["sigma"]

    z_low = model.truncation_normalizer(mu - 3 * sigma, **params)
    z_high = model.truncation_normalizer(mu + 12 * sigma, **params)

    assert 0.0 < z_low <= 1.0
    assert 0.0 < z_high <= 1.0
    # Raising the upper cut never decreases the captured mass.
    assert z_high >= z_low
    # With the cut far above the mode, ~all mass is below it.
    assert z_high == pytest.approx(1.0, abs=0.02)


def test_gaussian_pdf_matches_scipy_norm():
    model = GaussianModel()
    params = {"mu": 50.0, "sigma": 5.0}
    grid = np.linspace(30, 70, 200)
    from scipy import stats

    expected = stats.norm.pdf(grid, loc=50.0, scale=5.0)
    assert np.allclose(model.pdf_untruncated(grid, **params), expected)


def test_gaussian_tail_probability_matches_quad():
    model = GaussianModel()
    params = {"mu": 50.0, "sigma": 5.0}
    u0 = 60.0
    expected, _ = integrate.quad(
        lambda x: float(model.pdf_untruncated(np.array([x]), **params)[0]), u0, np.inf
    )
    assert model.tail_probability(u0, **params) == pytest.approx(expected, rel=1e-4)


def test_tail_support_capped_at_z_upper(model_and_params):
    """The tail support is truncated at z = TAIL_SUPPORT_Z_UPPER: any threshold
    at or beyond that z has exactly zero tail probability, for every model."""
    from opennta.tracking.trackmate.fitting_models import TAIL_SUPPORT_Z_UPPER

    model, params = model_and_params
    mu, sigma = params["mu"], params["sigma"]
    u_cap = mu + TAIL_SUPPORT_Z_UPPER * sigma

    assert model.tail_probability(u_cap, **params) == pytest.approx(0.0, abs=1e-9)
    assert model.tail_probability(u_cap + 0.5 * sigma, **params) == 0.0


def test_tail_uses_capped_normalization(model_and_params):
    """Every model normalizes its tail over the same support (-inf, z=8], so
    tail = (F(8) - F(z0)) / F(8). Deep in the lower tail essentially all of the
    (capped) background mass lies above the threshold, giving tail ~ 1 -- this
    is the shared convention that makes alpha comparable across models, instead
    of Gaussian integrating to +inf while CS210 stopped at z=8."""
    model, params = model_and_params
    mu, sigma = params["mu"], params["sigma"]
    assert model.tail_probability(mu - 6 * sigma, **params) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_validate_params_rejects_nonpositive_sigma(model_id):
    cls, params = MODEL_PARAMS[model_id]
    model = cls()
    good = dict(params)
    bad = dict(params)
    bad["sigma"] = 0.0
    assert model.validate_params(good) is True
    assert model.validate_params(bad) is False
