"""Checks of the Moffat PSF model: one PSF, halo and spikes included, for stars and galaxies alike."""

import contextlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from mock_lsst_image_generation import IMAGE_CONFIG, generate_catalogue_images
from mock_lsst_image_generation.catalogue_pipeline import MockCatalogue
from mock_lsst_image_generation.coadd_synthesis import coadd_psf, single_coadd
from mock_lsst_image_generation.galaxy_rendering import render_catalogue
from mock_lsst_image_generation.psf import coadd_psf_image, moffat_core, wings_kernel

BANDS = IMAGE_CONFIG["bands"]
CFG = {**IMAGE_CONFIG, "psf_model": "moffat", "canvas_npix": 401}
STAR_XY, GALAXY_XY = (150.0, 200.0), (300.0, 200.0)


@pytest.fixture(scope="module")
def coadd():
    return single_coadd(0, CFG, epoch="1y")["1y_moffat"]


@pytest.fixture(scope="module")
def catalogue():
    """A 12th magnitude star and a bulge-only galaxy, far enough apart that their cores do not overlap."""
    star = dict(type="star", x_pix=STAR_XY[0], y_pix=STAR_XY[1], mag_r_total=12.0, saturation_r=1e4,
                **{f"flux_{band}_total": 10 ** (-0.4 * (12.0 - 31.4)) for band in BANDS})
    galaxy = dict(type="passive", x_pix=GALAXY_XY[0], y_pix=GALAXY_XY[1], pa_deg=30.0, re_bulge_arcsec=0.6,
                  ellipticity_bulge=0.2, n_bulge=4.0, re_disc_arcsec=np.nan, ellipticity_disc=np.nan, n_disc=np.nan,
                  **{f"flux_{band}_bulge": 1e5 for band in BANDS}, **{f"flux_{band}_disc": np.nan for band in BANDS})
    empty = pd.DataFrame()
    return MockCatalogue(pd.DataFrame([galaxy, star]), empty, empty, empty)


@pytest.fixture(scope="module")
def images(catalogue, coadd):
    """(base render at the Moffat core, the same taken to the coadd PSF)."""
    with contextlib.redirect_stderr(io.StringIO()):
        base = render_catalogue(catalogue, {band: moffat_core(coadd, band) for band in BANDS}, CFG)[0]
    return base, coadd_psf(base, coadd, CFG)


def test_moffat_coadd_settings(coadd):
    again = single_coadd(0, CFG, epoch="1y")["1y_moffat"]
    assert (again["fwhm_arcsec"], again["moffat_beta"]) == (coadd["fwhm_arcsec"], coadd["moffat_beta"])
    for band in BANDS:
        assert coadd["fwhm_arcsec"][band] == pytest.approx(CFG["moffat_fwhm"][band], rel=0.15)
        assert coadd["moffat_beta"][band] == pytest.approx(CFG["moffat_beta"][band], rel=0.25)
    assert min(coadd["fwhm_arcsec"], key=coadd["fwhm_arcsec"].get) in "ri"
    other = single_coadd(1, CFG, epoch="1y")["1y_moffat"]
    assert other["fwhm_arcsec"] != coadd["fwhm_arcsec"]  # each catalogue gets its own jitter


def test_wings_kernel_is_halo_and_spikes(coadd):
    for band in BANDS:
        kernel, spikes = wings_kernel(band, coadd, CFG)
        half = CFG["psf_wings"]["half_pix"]
        assert kernel.sum() == pytest.approx(1.0) and kernel.min() >= 0
        assert 0.9 < kernel[half, half] < 1 and 0 < spikes.sum() < 0.05
        assert np.array_equal(kernel, wings_kernel(band, coadd, CFG)[0])


def test_star_is_its_flux_times_the_saved_psf(catalogue, coadd, images):
    psf = coadd_psf_image(coadd, CFG)
    half = CFG["psf_stamp_pix"] // 2
    x, y = int(STAR_XY[0]), int(STAR_XY[1])
    for i, band in enumerate(BANDS):
        flux = catalogue.galaxies[f"flux_{band}_total"].iloc[1]
        star = images[1][i, y - half:y + half + 1, x - half:x + half + 1] / flux
        assert 0.9 < psf[i].sum() < 1
        assert np.abs(star - psf[i]).max() < 1e-4 * psf[i].max()


def test_galaxies_get_the_halo(images):
    base, clean = images
    yy, xx = np.mgrid[:base.shape[1], :base.shape[2]]
    far = (np.hypot(xx - GALAXY_XY[0], yy - GALAXY_XY[1]) > 25) & (xx > 225)  # > 5" out, away from the star
    near = np.hypot(xx - GALAXY_XY[0], yy - GALAXY_XY[1]) < 100
    r = BANDS.index("r")
    assert clean[r][far & near].sum() > 2 * base[r][far & near].sum()


def test_moffat_needs_single_mode(catalogue, tmp_path):
    with pytest.raises(ValueError, match="psf_model"):
        generate_catalogue_images(catalogue, "train", tmp_path, "training", cfg=CFG)


def test_moffat_coadd_rebuilds_from_its_manifest(catalogue, coadd, images, tmp_path):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        generate_catalogue_images(catalogue, "train", tmp_path, "single", cfg=CFG, epoch="1y")
    out = tmp_path / "train"
    entry = json.loads((out / "coadd_manifest.json").read_text())["combos"]["1y_moffat"]
    assert json.loads((out / "base_meta.json").read_text())["psf_model"] == "moffat"
    assert np.array_equal(np.load(out / "train_1y_moffat_psf.npy"), coadd_psf_image(coadd, CFG))
    assert np.array_equal(coadd_psf(images[0], entry, CFG), images[1])
    spikes = np.load(out / "base_spikes_signal.npy")
    assert 0 < spikes.sum() < 0.05 * images[1].sum()
