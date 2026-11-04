"""Checks of catalogue regression statistics, physical consistency and same-platform reproducibility.

The regression test pins the seed-42, 1000 x 1000 px realisation built from the shipped COSMOS subset. If you change
the model on purpose, rerun it, check the new catalogue is what you intended, and update REFERENCE.
"""

import contextlib
import io

import numpy as np
import pytest
from scipy.stats import poisson

from mock_lsst_image_generation.config import OUT_BANDS
from mock_lsst_image_generation import (CONFIG, DEFAULT_COSMOS_PATH, build_mock_catalogue, catalogue_splits,
                                        generate_mock_catalogues, load_cosmos2025_catalogue)
from mock_lsst_image_generation.sfregion_generation import target_sfregion_light_fraction
from mock_lsst_image_generation.utils import resolved_mask

# Counts and column sums of the reference realisation (macOS, pinned requirements). Other platforms can differ in the
# last few decimal places, which can flip galaxies across cuts. Star-forming region counts are tested against their
# distribution
# separately because changed hosts and rejection sampling can shift the subsequent random draws.
REFERENCE_COUNTS = dict(galaxies=4254, tidal_pairs=2, donors=18476)
REFERENCE_SUMS = dict(z=9245.306949574317, logM=33968.64030832245, mag_r_total=118973.79265390243)
# Sizes vary far more between donors than magnitudes, so platform-level differences in which donor a galaxy draws move
# the summed Re by ~1% (seen on Linux CI); the median size is compared instead, with a looser tolerance.
REFERENCE_MEDIAN_RE_ARCSEC = 0.2014022695138109
BLOBS_PER_PAIR = 162  # rows of the tidal table (one per blob and band) made for each pair
COUNT_TOLERANCE, SUM_TOLERANCE, MEDIAN_RE_TOLERANCE = 0.01, 0.01, 0.03


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def cosmos():
    return load_cosmos2025_catalogue()


@pytest.fixture(scope="module")
def catalogue(cosmos):
    return quiet(build_mock_catalogue, cosmos, dict(seed=42, npix=1000, stars=False))  # galaxies only; see test_stars


def test_shipped_cosmos_subset_loads(cosmos):
    assert DEFAULT_COSMOS_PATH.exists()
    assert len(cosmos) == 87286
    assert {"z", "logM", "logsSFR", "m_r"}.issubset(cosmos.columns)


def test_catalogue_splits_are_stable():
    assert catalogue_splits(4) == dict(train=42, valid=101, calib=202, test=303)
    assert catalogue_splits(6) == {**catalogue_splits(4), "extra1": 404, "extra2": 505}
    assert catalogue_splits(3, numbered=True) == {"1": 42, "2": 101, "3": 202}
    with pytest.raises(ValueError):
        catalogue_splits(0)


def test_catalogue_matches_reference(catalogue):
    tables = catalogue._asdict()
    for name, ref in REFERENCE_COUNTS.items():
        assert len(tables[name]) == pytest.approx(ref, rel=COUNT_TOLERANCE, abs=2), name
    # Whether a pair qualifies can flip across platforms, so blobs are checked per pair rather than as a total.
    assert len(tables["tidal_blobs"]) == BLOBS_PER_PAIR * len(tables["tidal_pairs"])
    for col, ref in REFERENCE_SUMS.items():
        assert catalogue.galaxies[col].sum() == pytest.approx(ref, rel=SUM_TOLERANCE), col
    assert catalogue.galaxies["re_total_arcsec"].median() == pytest.approx(REFERENCE_MEDIAN_RE_ARCSEC,
                                                                          rel=MEDIAN_RE_TOLERANCE)


def test_catalogue_is_physically_sane(catalogue):
    g = catalogue.galaxies
    npix = 1000
    assert g["id"].tolist() == list(range(len(g)))
    assert g["x_pix"].between(0, npix).all() and g["y_pix"].between(0, npix).all()
    assert g["z"].between(CONFIG["z_min"], CONFIG["z_max"]).all()
    main = g["lsb_population"].isna() if "lsb_population" in g else np.ones(len(g), bool)
    assert g.loc[main, "logM"].between(CONFIG["logm_min"], CONFIG["logm_max"]).all()
    assert g.loc[~main, "logM"].between(5.0, CONFIG["logm_max"]).all()  # added LSB galaxies may be lighter
    for band in OUT_BANDS:
        assert np.isfinite(g[f"mag_{band}_total"]).all() and (g[f"flux_{band}_total"] > 0).all(), band
    assert (g["sb_r_total"] <= CONFIG["render_mu_r_max"]).all()
    assert set(catalogue.sfregions["parent_id"]).issubset(g["id"])


def test_sfregion_counts_follow_host_population(catalogue):
    # Rejection sampling and upstream cuts can change later random draws across platforms.
    # Check the capped Poisson population against its model, not one machine's realisation.
    g = catalogue.galaxies
    re = g["re_disc_arcsec_before_sfregions"]
    hosts = g.loc[g["type"].eq("star_forming") & resolved_mask(g, CONFIG) & np.isfinite(re) & (re > 0)]
    rate = (target_sfregion_light_fraction(hosts["logM"], hosts["logSFR"]).to_numpy()
            / CONFIG["sfregion_mean_single_u_frac"])
    k = np.arange(1, CONFIG["sfregion_n_max_per_gal"] + 1)[:, None]
    survival = poisson.sf(k - 1, rate)
    mean = survival.sum(axis=0)
    variance = ((2 * k - 1) * survival).sum(axis=0) - mean**2
    assert abs(len(catalogue.sfregions) - mean.sum()) <= 6 * np.sqrt(variance.sum())
    counts = catalogue.sfregions.groupby("parent_id").size().reindex(g["id"], fill_value=0)
    np.testing.assert_array_equal(counts.to_numpy(), g["N_sfregions"].to_numpy())
    assert counts.between(0, CONFIG["sfregion_n_max_per_gal"]).all()
    assert set(catalogue.sfregions["parent_id"]).issubset(hosts["id"])
    for band in OUT_BANDS:
        flux = catalogue.sfregions.groupby("parent_id")[f"flux_{band}_sfregion"].sum().reindex(g["id"], fill_value=0)
        np.testing.assert_allclose(flux.to_numpy(), g[f"flux_{band}_sfregion"].to_numpy())
        np.testing.assert_allclose(g[f"flux_{band}_disc_before_sfregions"],
                                   g[f"flux_{band}_disc"] + g[f"flux_{band}_sfregion"] + g[f"flux_{band}_tidal"])


def test_same_seed_same_catalogue_different_seed_differs(cosmos, catalogue):
    again = quiet(build_mock_catalogue, cosmos, dict(seed=42, npix=1000, stars=False))
    for name in ("galaxies", "sfregions", "tidal_pairs", "tidal_blobs"):
        assert getattr(again, name).equals(getattr(catalogue, name)), name
    other = quiet(build_mock_catalogue, cosmos, dict(seed=7, npix=300, stars=False)).galaxies
    assert not other.equals(catalogue.galaxies)
    # Smaller frames: these only check difference and reproducibility, not the pinned seed-42 reference.
    with_stars = [quiet(build_mock_catalogue, cosmos, dict(seed=42, npix=300)).galaxies for _ in range(2)]
    assert with_stars[0].equals(with_stars[1])  # star rows are reproducible too


def test_generate_writes_all_csvs(cosmos, tmp_path):
    quiet(generate_mock_catalogues, cosmos, 1, tmp_path, cfg=dict(npix=300))
    stem = tmp_path / "mock_catalogue_train"
    for suffix in ["", "_sfregions", "_tidal", "_tidal_pairs"]:
        assert (tmp_path / f"{stem.name}{suffix}.csv").exists(), suffix


def test_low_redshift_mass_function_matches_gama():
    """Anchored locally to Baldry et al. 2012 (GAMA): exact at z <= gsmf_z_eval_min, unchanged above the fade-out."""
    from mock_lsst_image_generation.gsmf_sampling import gama_gsmf, quiescent_gsmf, star_forming_gsmf
    logm = np.linspace(7.0, 11.5, 10)
    local = star_forming_gsmf(logm, 0.1) + quiescent_gsmf(logm, 0.1)
    np.testing.assert_allclose(local, gama_gsmf(logm), rtol=1e-6)
    high = star_forming_gsmf(logm, 1.0) + quiescent_gsmf(logm, 1.0)
    unanchored = star_forming_gsmf(logm, 1.0, anchored=False) + quiescent_gsmf(logm, 1.0, anchored=False)
    np.testing.assert_allclose(high, unanchored)
