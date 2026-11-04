"""Checks of the truth table of peak-like components and of which components are peaks in a coadd."""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from mock_lsst_image_generation import IMAGE_CONFIG, MockCatalogue, build_mock_catalogue, load_cosmos2025_catalogue
from mock_lsst_image_generation.coadd_synthesis import single_coadd
from mock_lsst_image_generation.galaxy_rendering import render_catalogue, split_stars
from mock_lsst_image_generation.truth_peaks import KINDS, find_truth_peaks, truth_table

BANDS = IMAGE_CONFIG["bands"]


@pytest.fixture(scope="module")
def catalogue():
    with contextlib.redirect_stdout(io.StringIO()):
        return build_mock_catalogue(load_cosmos2025_catalogue(), dict(seed=42, npix=400))


@pytest.fixture(scope="module")
def flagged(catalogue):
    """(truth, peaks) for the 10-year nominal coadd of a fixed 400 x 400 px canvas."""
    cfg = {**IMAGE_CONFIG, "canvas_npix": 400}
    with contextlib.redirect_stderr(io.StringIO()):
        image, origin = render_catalogue(catalogue, {band: cfg["base_fwhm"] for band in BANDS}, cfg, star_seed=0)
    coadd = single_coadd(0, cfg, epoch="10y")["10y_nominal"]
    return find_truth_peaks(truth_table(catalogue, BANDS, cfg["pixscale"]), image, origin, coadd, cfg)


def test_truth_table_lists_every_component(catalogue):
    truth = truth_table(catalogue, BANDS, IMAGE_CONFIG["pixscale"])
    galaxies, stars = split_stars(catalogue.galaxies)
    tidal = catalogue.tidal_blobs
    n_tidal = len(tidal[["tidal_pair_id", "tidal_blob_id"]].drop_duplicates()) if len(tidal) else 0
    counts = truth["kind"].value_counts()
    assert set(counts.index) <= set(KINDS)
    assert counts.get("galaxy", 0) == len(galaxies) and counts.get("star", 0) == len(stars)
    assert counts.get("sfregion", 0) == len(catalogue.sfregions) and counts.get("tidal", 0) == n_tidal
    assert truth["truth_id"].tolist() == list(range(len(truth)))
    assert np.isfinite(truth[["x_pix", "y_pix", *[f"flux_{band}" for band in BANDS]]]).all().all()
    sfr = truth["kind"] == "sfregion"
    assert truth.loc[sfr, "parent_id"].isin(galaxies["id"]).all() and (truth.loc[~sfr, "kind"] != "sfregion").all()
    assert np.isclose(truth.loc[sfr, "flux_r"].sum(), catalogue.sfregions["flux_r_sfregion"].sum())


def test_bright_peaks_belong_to_components(flagged):
    _, peaks = flagged
    bright = peaks[peaks["snr"] >= 5]
    assert len(bright) > 50
    assert (bright["truth_id"] < 0).mean() < 0.05  # almost every clear maximum is a catalogued component
    assert (peaks["match_dist"].dropna() <= 3.0).all()


def test_peak_flags_follow_the_peak_table(flagged):
    truth, peaks = flagged
    for band in BANDS:
        owned = peaks[(peaks["band"] == band) & (peaks["truth_id"] >= 0)]
        assert set(truth.loc[truth[f"peak_{band}"], "truth_id"]) == set(owned["truth_id"])
        assert truth[f"n_peaks_{band}"].sum() == len(owned)
        assert truth.loc[~truth[f"peak_{band}"], f"peak_snr_{band}"].isna().all()
    regular = truth[(truth["kind"] == "galaxy") & (truth["population"] == "") & truth["in_image"]]
    assert regular.nlargest(1, "flux_r").iloc[0]["peak_r"]  # the brightest ordinary galaxy is a peak
    in_image_stars = truth[(truth["kind"] == "star") & truth["in_image"]]
    assert in_image_stars["peak_r"].mean() > 0.8  # stars are points: nearly all are r-band peaks


def test_tidal_blobs_become_one_row_each_with_their_donor():
    """Tidal blobs are stored one row per blob and band; the truth table has one row per blob."""
    flux = {f"flux_{band}_total": 10.0 for band in BANDS}
    galaxies = pd.DataFrame(dict(id=[0, 1], type="star_forming", x_pix=[10.0, 30.0], y_pix=[10.0, 10.0],
                                 hubble_type=["Sb", "Irr"], **flux))
    tidal = pd.DataFrame([dict(tidal_pair_id=0, tidal_blob_id=k, donor_id=1, gal1_id=0, gal2_id=1, band=band,
                               x_pix_tidal=10.0 + 2 * k, y_pix_tidal=10.0, sigma_pix=2.0, flux_tidal=0.5)
                          for k in range(3) for band in BANDS])
    truth = truth_table(MockCatalogue(galaxies, pd.DataFrame(), tidal, pd.DataFrame()), BANDS, 0.2)
    blobs = truth[truth["kind"] == "tidal"]
    assert truth["kind"].tolist() == ["galaxy"] * 2 + ["tidal"] * 3
    assert blobs["component_index"].tolist() == [0, 1, 2] and (blobs["parent_id"] == 1).all()
    assert (blobs["hubble_type"] == "Irr").all() and (blobs["flux_r"] == 0.5).all()
