"""BCG-like sources: a core-Sersic envelope (BCG core + extended diffuse light), from ICL_DoubleSersic_Injection.ipynb.

Profile (Graham et al. 2003, eq. 1):
    I(R) = I' (1 + (Rb / R)^alpha)^(gamma / alpha) exp(-b_n ((R^alpha + Rb^alpha) / Re^alpha)^(1 / (n alpha)))
with n = 3.9, Rb / Re = 3.63 / 30 (Abell 2261), gamma = 0.02, alpha = 3.6 and axis ratio 0.7, as in the notebook.
Photometry also follows the notebook: mu_r is the mean r-band surface brightness within the half-light area of the
PSF-convolved image, and the other bands follow fixed red colours (u-r 2.2, g-r 0.65, r-i 0.35, i-z 0.25, z-y 0.10).
A few per frame (cfg['bcg_per_deg2']) are added at random positions, with Re and mu_r drawn from cfg['bcg_re_arcsec']
(log-uniform) and cfg['bcg_mu_r'] (uniform). Rows have profile = 'core_sersic'; galaxy_rendering draws them here.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .photometry import mag_to_flux_njy
from .psf import convolve_psf

CORE_SERSIC = dict(n=3.9, rb_over_re=3.63 / 30.0, gamma=0.02, alpha=3.6, q=0.7)
COLOURS_FROM_R = dict(u=2.2, g=0.65, r=0.0, i=-0.35, z=-0.60, y=-0.70)  # m_band - m_r (notebook colours)
STAMP_RE = 6.0  # stamps reach this many Re


def core_sersic_image(re_pix, half, pa_rad, p=CORE_SERSIC):
    """Unit-sum core-Sersic image on a (2 half + 1)^2 grid centred on the source (notebook's core_sersic_2d)."""
    n, rb = p["n"], p["rb_over_re"] * re_pix
    bn = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n ** 2)
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    xp, yp = x * np.cos(pa_rad) + y * np.sin(pa_rad), -x * np.sin(pa_rad) + y * np.cos(pa_rad)
    r = np.maximum(np.hypot(xp, yp / p["q"]), 0.5)
    a, g = p["alpha"], p["gamma"]
    prof = (1.0 + (rb / r) ** a) ** (g / a) * np.exp(-bn * ((r ** a + rb ** a) / re_pix ** a) ** (1.0 / (n * a)))
    return prof / prof.sum()


def half_light_area_arcsec2(image, pixscale):
    """Area of the brightest pixels holding half the flux (the notebook's definition behind mu_r)."""
    flat = np.sort(np.clip(image, 0, None).ravel())[::-1]
    return (np.searchsorted(np.cumsum(flat), 0.5 * flat.sum()) + 1) * pixscale ** 2


def add_bcgs(mock, cfg, rng):
    """Catalogue with a Poisson number of BCG-like sources appended (profile 'core_sersic', ids continuing)."""
    area = (cfg["npix"] * cfg["pixscale"] / 3600.0) ** 2
    n = rng.poisson(cfg["bcg_per_deg2"] * area)
    if n == 0:
        return mock
    re = 10 ** rng.uniform(*np.log10(cfg["bcg_re_arcsec"]), n)
    mu_r = rng.uniform(*cfg["bcg_mu_r"], n)
    pa = rng.uniform(0, 180, n)
    t = pd.DataFrame(dict(id=len(mock) + np.arange(n), x_pix=rng.uniform(0, cfg["npix"], n),
                          y_pix=rng.uniform(0, cfg["npix"], n), z=rng.uniform(0.05, 0.4, n), logM=rng.uniform(11.3, 12.0, n),
                          type="passive", kind="pa", hubble_type="cD", profile="core_sersic", lsb_population="bcg",
                          pa_deg=pa, re_total_arcsec=re, ellipticity_total=1 - CORE_SERSIC["q"], bcg_mu_r=mu_r))
    psf_r = cfg["psf_fwhm_arcsec"]["r"] / 2.355 / cfg["pixscale"]
    mag_r = np.empty(n)
    for i in range(n):  # mu_r -> r magnitude through the convolved half-light area, as in the notebook
        re_pix = re[i] / cfg["pixscale"]
        img = gaussian_filter(core_sersic_image(re_pix, int(STAMP_RE * re_pix), np.radians(pa[i])), psf_r)
        mag_r[i] = mu_r[i] - 2.5 * np.log10(half_light_area_arcsec2(img, cfg["pixscale"]))
    for band in cfg["out_bands"]:
        t[f"mag_{band}_total"] = mag_r + COLOURS_FROM_R[band]
        t[f"flux_{band}_total"] = mag_to_flux_njy(t[f"mag_{band}_total"])
    t["sb_r_total"] = mu_r
    assert mock["id"].tolist() == list(range(len(mock)))
    print(f"Added {n} BCG-like core-Sersic sources", flush=True)
    return pd.concat([mock, t], ignore_index=True)


def render_bcg(record, psf, bands, pixscale):
    """(xmin, ymin, cube) stamp of a core-Sersic record, each band convolved with its PSF (psf.convolve_psf)."""
    re_pix = float(record["re_total_arcsec"]) / pixscale
    half = int(STAMP_RE * re_pix)
    ix, iy = int(np.floor(record["x_img"] + 0.5)), int(np.floor(record["y_img"] + 0.5))
    unit = core_sersic_image(re_pix, half, np.radians(float(record["pa_deg"])))
    cube = np.zeros((len(bands), 2 * half + 1, 2 * half + 1), np.float32)
    for k, band in enumerate(bands):
        flux = float(record.get(f"flux_{band}_total", 0.0))
        if np.isfinite(flux) and flux > 0:
            cube[k] = convolve_psf(unit * flux, psf[band], pixscale, mode="reflect")
    return ix - half, iy - half, cube

