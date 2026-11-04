"""Turn one noise-free render into LSST-like coadds at any survey depth and seeing.

Coadding is linear, so a coadd is the scene convolved with the visit-averaged PSF plus the stacked noise. The scene is
therefore rendered once at a narrow base PSF (the expensive GalSim step), and each (depth, seeing) combination is made
from it by broadening to the target PSF (Gaussian FWHMs add in quadrature) and adding noise for its number of visits.
Depth and seeing are independent axes of the grid, so a detector trained on it cannot learn a spurious link between
them. single_coadd makes one coadd at a chosen depth and PSF instead of the whole grid.
"""

from typing import TypedDict

import numpy as np
from scipy.ndimage import gaussian_filter

from .psf import add_wings
from .stars import saturate_stars

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
# Bits of the saved coadd mask (uint8, band x y x), named as the LSST stack's mask planes so they can be copied across.
MASK_PLANES = {"SAT": 0}


class CoaddSettings(TypedDict):
    """Settings for one coadd, with per-band mappings distinguished from scalar metadata."""

    epoch: str | None
    survey_fraction: float
    f_r: float
    n_visit: dict[str, int]
    fwhm_arcsec: dict[str, float]
    seed_entropy: list[int]
    noise_seed: np.random.SeedSequence
    psf_model: str
    moffat_beta: dict[str, float] | None
    psf_wings: dict | None


def visits_at(survey_fraction, cfg):
    """Visits per band accumulated after this fraction of the 10-year survey (at least one)."""
    return {band: max(1, int(round(cfg["visits_10yr"][band] * survey_fraction))) for band in cfg["bands"]}


def jittered_band_fwhm(fwhm_r, seed, cfg):
    """Per-band PSF FWHM for one seeing value: the r-band FWHM scaled by the nominal band ratios, each with a small
    random jitter, and kept above the base PSF so the broadening kernel exists."""
    rng = np.random.default_rng(seed)
    nominal = cfg["nominal_fwhm"]
    fwhm = {}
    for band in cfg["bands"]:
        jitter = float(rng.normal(0.0, cfg["fwhm_ratio_jitter"]))
        fwhm[band] = float(max(fwhm_r * (nominal[band] / nominal["r"]) * (1.0 + jitter), cfg["base_fwhm"] + 0.02))
    return fwhm


def coadd_grid(catalogue_index, cfg) -> dict[str, CoaddSettings]:
    """Every (epoch, seeing) combination for one catalogue, as {key: settings}, with key e.g. '1y_fwhm110'.

    Each combination's jitter and noise come from its own seed, SeedSequence([catalogue_index, epoch index, seeing
    index]), so any coadd can be rebuilt on its own (e.g. on the fly during training) and is identical every time.
    """
    grid: dict[str, CoaddSettings] = {}
    for epoch_index, (epoch, survey_fraction) in enumerate(cfg["epochs"].items()):
        for fwhm_index, fwhm_r in enumerate(cfg["fwhm_grid_r"]):
            seed_entropy = [catalogue_index, epoch_index, fwhm_index]
            jitter_seed, noise_seed = np.random.SeedSequence(seed_entropy).spawn(2)
            grid[f"{epoch}_fwhm{int(round(fwhm_r * 100)):03d}"] = CoaddSettings(
                epoch=epoch, survey_fraction=survey_fraction, f_r=fwhm_r, n_visit=visits_at(survey_fraction, cfg),
                fwhm_arcsec=jittered_band_fwhm(fwhm_r, jitter_seed, cfg), seed_entropy=seed_entropy,
                noise_seed=noise_seed, psf_model="gaussian", moffat_beta=None, psf_wings=None)
    return grid


EPOCH_CHOICES = [f"{n}m" for n in range(1, 12)] + [f"{n}y" for n in range(1, 11)]


def survey_fraction(epoch):
    """Fraction of the 10-year survey for an epoch in EPOCH_CHOICES: '10y' -> 1.0, '1y' -> 0.1, '6m' -> 0.05."""
    if epoch not in EPOCH_CHOICES:
        raise ValueError(f"Unknown epoch {epoch!r}; choose from {', '.join(EPOCH_CHOICES)}")
    return int(epoch[:-1]) / (10.0 if epoch.endswith("y") else 120.0)


def single_coadd(catalogue_index, cfg, epoch=None, n_exp=None, psf_fwhm=None) -> dict[str, CoaddSettings]:
    """One coadd, in the same form as a coadd_grid entry, given either

      epoch             e.g. '10y': that epoch's visits, at the nominal PSF in every band; or
      n_exp, psf_fwhm   r-band visits and r-band PSF FWHM (arcsec); the other bands scale by the 10-year visit ratios
                        and the nominal PSF ratios.

    The noise seed depends only on the catalogue and the coadd's visits and PSF, so the same request gives the same
    image every time. With cfg['psf_model'] 'moffat', the PSF is a Moffat scaled from moffat_fwhm, and each band's
    FWHM and Moffat index get a random jitter drawn from that seed.
    """
    moffat = cfg.get("psf_model", "gaussian") == "moffat"
    nominal = cfg["moffat_fwhm"] if moffat else cfg["nominal_fwhm"]
    if epoch is not None and n_exp is None and psf_fwhm is None:
        key, fraction, fwhm = f"{epoch}_{'moffat' if moffat else 'nominal'}", survey_fraction(epoch), dict(nominal)
    elif epoch is None and n_exp is not None and psf_fwhm is not None:
        key, fraction = f"nexp{int(n_exp)}_fwhm{int(round(psf_fwhm * 100)):03d}", n_exp / cfg["visits_10yr"]["r"]
        fwhm = {band: psf_fwhm * nominal[band] / nominal["r"] for band in cfg["bands"]}
    else:
        raise ValueError("give either epoch alone, or both n_exp and psf_fwhm")
    if fraction <= 0 or (not moffat and min(fwhm.values()) <= cfg["base_fwhm"]):
        raise ValueError(f"need at least one visit and a PSF wider than the {cfg['base_fwhm']}\" base PSF in every band")
    n_visit = visits_at(fraction, cfg)
    seed_entropy = [catalogue_index, 999, n_visit["r"], int(round(fwhm["r"] * 1000))]  # never matches a grid seed
    beta = None
    if moffat:
        # Rounded as the manifest stores them, so the PSF rebuilt from it is the one the image was made with.
        rng = np.random.default_rng(seed_entropy + [6])
        fwhm = {band: round(fwhm[band] * (1 + rng.normal(0, cfg["fwhm_ratio_jitter"])), 4) for band in cfg["bands"]}
        beta = {band: round(cfg["moffat_beta"][band] * (1 + rng.normal(0, cfg["moffat_beta_jitter"])), 4)
                for band in cfg["bands"]}
    return {key: CoaddSettings(epoch=epoch, survey_fraction=fraction, f_r=fwhm["r"], n_visit=n_visit, fwhm_arcsec=fwhm,
                      seed_entropy=seed_entropy, noise_seed=np.random.SeedSequence(seed_entropy),
                      psf_model="moffat" if moffat else "gaussian", moffat_beta=beta,
                      psf_wings=dict(cfg["psf_wings"]) if moffat else None)}


def broaden_band(band_image, band, target_fwhm, cfg):
    """Broaden one band's base-PSF image (y, x) to that band's target FWHM (arcsec) with a Gaussian of the
    quadrature difference."""
    base_fwhm, target = cfg["base_fwhm"], float(target_fwhm[band])
    if target < base_fwhm:
        raise ValueError(f"target FWHM {target:.3f} < base {base_fwhm:.3f} for band {band}")
    sigma_pix = (np.sqrt(target ** 2 - base_fwhm ** 2) * FWHM_TO_SIGMA) / cfg["pixscale"]
    if sigma_pix <= 0:
        return band_image.copy()
    return gaussian_filter(band_image, sigma=sigma_pix, mode="constant", cval=0.0, truncate=5.0)


def broaden(image, target_fwhm, cfg):
    """Broaden a base-PSF image to the per-band target FWHM (arcsec) with a Gaussian of the quadrature difference."""
    broadened = np.empty_like(image)
    for i, band in enumerate(cfg["bands"]):
        broadened[i] = broaden_band(image[i], band, target_fwhm, cfg)
    return broadened


def coadd_psf_band(band_image, band, coadd, cfg):
    """One band of a base render taken to a coadd's PSF: broadened (Gaussian model) or convolved with the halo and
    spikes kernel (Moffat model, whose base render already has the coadd's core; psf.py)."""
    if coadd.get("psf_model", "gaussian") == "moffat":
        return add_wings(band_image, band, coadd, cfg)
    return broaden_band(band_image, band, coadd["fwhm_arcsec"], cfg)


def coadd_psf(image, coadd, cfg):
    """A base render taken to a coadd's PSF in every band (coadd_psf_band)."""
    clean = np.empty_like(image)
    for i, band in enumerate(cfg["bands"]):
        clean[i] = coadd_psf_band(image[i], band, coadd, cfg)
    return clean


def sky_sigma(band, n_visit, cfg):
    """Per-pixel sky noise (nJy) of an n_visit coadd: the 10-year value, from the 5-sigma point-source depth over a
    nominal-PSF aperture, scaled by sqrt(10-year visits / n_visit). It depends on depth only, never on the seeing."""
    n_pix_psf = 1.13 * (cfg["nominal_fwhm"][band] / cfg["pixscale"]) ** 2
    sigma_10yr = 10 ** (-0.4 * (cfg["depth_10yr"][band] - cfg["zeropoint"])) / (5 * np.sqrt(n_pix_psf))
    return sigma_10yr * np.sqrt(cfg["visits_10yr"][band] / n_visit)


def add_noise(image, n_visit, seed, cfg):
    """Noisy coadd and its variance: sky Gaussian noise plus source Poisson noise averaged over n_visit visits."""
    rng = np.random.default_rng(seed)
    signal, variance = image.copy(), np.zeros_like(image)
    for i, band in enumerate(cfg["bands"]):
        visits = int(n_visit[band])
        sky = sky_sigma(band, visits, cfg)
        source = np.maximum(image[i], 0)
        signal[i] += rng.poisson(visits * source).astype(np.float32) / visits - source
        signal[i] += rng.normal(0.0, sky, image[i].shape).astype(np.float32)
        variance[i] = sky ** 2 + source / visits
    return signal, variance


def synthesise_coadd(base_image, coadd, cfg, stars=None, origin=(0, 0), return_mask=False):
    """(signal, variance) float32 cubes for one coadd_grid / single_coadd entry, from the base-PSF render; with
    return_mask, (signal, variance, mask), where mask is a uint8 cube whose bits are given by MASK_PLANES.

    Stars (catalogue rows in frame pixels; origin is the frame pixel at base_image[:, 0, 0]) get saturated cores after
    the noise, as in real coadds; their pixels are the SAT bit of the mask. The signal and variance are identical with
    or without return_mask."""
    clean = coadd_psf(base_image, coadd, cfg)
    signal, variance = add_noise(clean, coadd["n_visit"], coadd["noise_seed"], cfg)
    saturated = np.zeros(signal.shape, bool)
    if stars is not None and len(stars):
        signal = saturate_stars(signal, clean, stars, origin, cfg["bands"],
                                np.random.default_rng(list(coadd["seed_entropy"]) + [3]), mask_out=saturated)
    signal, variance = signal.astype(np.float32, copy=False), variance.astype(np.float32, copy=False)
    if not return_mask:
        return signal, variance
    return signal, variance, (saturated.astype(np.uint8) << MASK_PLANES["SAT"])
