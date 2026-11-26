"""Unit tests for GeoChem-Analytics.

Run from the repo root with:  pytest -q
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

logging.disable(logging.WARNING)  # silence Streamlit "bare mode" warnings on import
import geochem_app as g  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "geochem_app.py"


@pytest.fixture
def comp():
    """Small random closed composition: 50 samples x 5 parts, rows sum to 1."""
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(rng.lognormal(size=(50, 5)), columns=list("ABCDE"))
    return g.close_rows(raw)


# --- Data cleaning --------------------------------------------------------

def test_below_detection_becomes_half_the_limit():
    out = g.coerce_assays(pd.Series(["<0.5", "2.0", "<0.01"]))
    assert out.tolist() == pytest.approx([0.25, 2.0, 0.005])


def test_closure_rows_sum_to_one(comp):
    assert np.allclose(comp.sum(axis=1), 1.0)


# --- Log-ratio transforms -------------------------------------------------

def test_clr_rows_sum_to_zero(comp):
    assert np.allclose(g.clr_transform(comp).sum(axis=1), 0.0)


def test_clr_is_scale_invariant(comp):
    """Multiplying a sample by a constant (e.g. % vs ppm) must not change CLR."""
    assert np.allclose(g.clr_transform(comp), g.clr_transform(comp * 1e4))


def test_alr_drops_reference_part(comp):
    out = g.alr_transform(comp, "E")
    assert out.shape == (50, 4)
    assert not any("ln(E/" in c for c in out.columns)


def test_ilr_has_d_minus_1_coords(comp):
    assert g.ilr_transform(comp).shape == (50, 4)


def test_ilr_preserves_aitchison_distance(comp):
    """ILR is an isometry: Euclidean distance in ILR space equals it in CLR space."""
    clr = g.clr_transform(comp).to_numpy()
    ilr = g.ilr_transform(comp).to_numpy()
    d_clr = np.linalg.norm(clr[0] - clr[1:], axis=1)
    d_ilr = np.linalg.norm(ilr[0] - ilr[1:], axis=1)
    assert np.allclose(d_clr, d_ilr)


# --- Geometry and multivariate helpers -----------------------------------

def test_ternary_corners():
    x, y = g.ternary_xy(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
    assert np.allclose(x, [0.0, 1.0, 0.5])
    assert np.allclose(y, [0.0, 0.0, np.sqrt(3) / 2])


def test_varimax_rotation_is_orthogonal():
    rng = np.random.default_rng(1)
    _, R = g.varimax(rng.normal(size=(8, 3)))
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)


def test_classical_mds_recovers_distances():
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(20, 2))
    D = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    emb = g.classical_mds(D, k=2)
    D2 = np.linalg.norm(emb[:, None] - emb[None], axis=-1)
    assert np.allclose(D, D2, atol=1e-8)


# --- Demo data and the full app ------------------------------------------

def test_demo_dataset_shape_and_bdl_values():
    df = g.make_demo()
    assert df.shape == (180, 13)
    assert df.loc[5, "Cu"] == "<0.5"


def test_app_runs_without_errors():
    """Smoke test: the whole Streamlit app renders on the demo data."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
