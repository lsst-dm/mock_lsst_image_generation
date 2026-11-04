"""Per-band PSFs: the profile every source is drawn with, and the Moffat model's halo and diffraction spikes.

A PSF is given per band either as a FWHM in arcsec (a Gaussian) or as a GalSim profile. In the Moffat model
(IMAGE_CONFIG psf_model) a coadd's PSF is core (x) K, where the core is a Moffat and K = (1 - f) delta + halo + spikes
is a unit-sum kernel of the stars' halo and spikes (STAR_CONFIG, with the per-star shape fixed by psf_wings).
Convolution is associative, so a scene drawn with the core and then convolved with K is the scene convolved with the
full PSF: galaxies get the halo and spikes stars get, and only the coadd step handles the wide kernel. K's spike angles,
arm brightness and halo streaks are drawn from the coadd's seed, so it can be rebuilt from a coadd_manifest.json entry.
"""

import galsim
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from .config import STAR_CONFIG
from .stars import BAND_WAVELENGTH_NM, pixel_grid, radial_streaks, spike_angles, spike_line


def psf_profile(psf):
    """GalSim profile of one band's PSF: a GSObject as given, or a Gaussian of that FWHM (arcsec)."""
    return psf if isinstance(psf, galsim.GSObject) else galsim.Gaussian(fwhm=psf)


def convolve_psf(image, psf, pixscale, mode="constant"):
    """A pixel-sampled image convolved with one band's PSF. A Gaussian (given as its FWHM) uses gaussian_filter with
    this boundary mode; a GalSim profile is sampled at pixel centres, like the image, and applied by FFT."""
    if not isinstance(psf, galsim.GSObject):
        return gaussian_filter(image, psf / 2.355 / pixscale, mode=mode)
    size = psf.getGoodImageSize(pixscale)
    size += 1 - size % 2  # odd, so mode="same" keeps the image centred
    kernel = psf.drawImage(nx=size, ny=size, scale=pixscale, method="no_pixel").array.astype(float)
    return fftconvolve(image, kernel / kernel.sum(), mode="same")


def moffat_core(coadd, band):
    """The Moffat core of one band of a Moffat-model coadd."""
    return galsim.Moffat(beta=coadd["moffat_beta"][band], fwhm=coadd["fwhm_arcsec"][band])


def wings_kernel(band, coadd, cfg, star_cfg=STAR_CONFIG):
    """(kernel, spikes) for one band of a Moffat-model coadd, both (2 half_pix + 1) square: K, which sums to 1, and
    its diffraction spikes alone."""
    wings, pixscale, fwhm = coadd["psf_wings"], cfg["pixscale"], coadd["fwhm_arcsec"][band]
    half = wings["half_pix"]
    size = 2 * half + 1
    angles = spike_angles(np.random.default_rng(list(coadd["seed_entropy"]) + [4]), cfg["bands"], star_cfg)[band]
    rng = np.random.default_rng(list(coadd["seed_entropy"]) + [5, cfg["bands"].index(band)])
    radius = np.hypot(*pixel_grid(half)) * pixscale
    streaks = radial_streaks(half, rng, wings["streak_strength"], pixscale)
    halos = [galsim.Moffat(beta=beta, fwhm=halo_fwhm).drawImage(nx=size, ny=size, scale=pixscale).array
             for beta, halo_fwhm in star_cfg["halo_basis"]]
    halo = sum(f * h for f, h in zip(star_cfg["halo_fraction"][band], halos))
    halo = halo * streaks / (1 + (radius / wings["halo_cut_arcsec"]) ** 6)
    # Spikes are spike_width core FWHMs across once convolved with the core, so they are narrower in K.
    width = np.sqrt(max(wings["spike_width"] ** 2 - 1, 0.01)) * fwhm
    core_arcsec = wings["spike_core_arcsec"] * BAND_WAVELENGTH_NM[band] / BAND_WAVELENGTH_NM["r"]
    spikes = star_cfg["spike_at_10arcsec"][band] * sum(
        weight * spike_line(half, angle, width, core_arcsec, wings["spike_slope"],
                            np.exp(rng.normal(0, star_cfg["arm_scatter"], 2)), pixscale, star_cfg["spike_taper_arcsec"])
        for angle, weight in angles)
    kernel = halo + spikes
    kernel[half, half] += 1 - kernel.sum()
    return kernel, spikes


def add_wings(band_image, band, coadd, cfg, spikes_only=False):
    """One band of a core-PSF image convolved with the coadd's K (or, with spikes_only, the light of its spikes)."""
    kernel = wings_kernel(band, coadd, cfg)[1 if spikes_only else 0]
    return fftconvolve(np.asarray(band_image, float), kernel, mode="same").astype(np.float32)


def coadd_psf_image(coadd, cfg):
    """(band, n, n) image of a coadd's PSF centred on the middle pixel, n = cfg['psf_stamp_pix'], with the pixel
    response the rendered sources have. It is the PSF cut out, not renormalised: each band sums to the fraction of a
    point source's light inside the stamp, which in the Moffat model leaves out most of the halo and spikes."""
    n, pixscale = cfg["psf_stamp_pix"], cfg["pixscale"]
    images = np.zeros((len(cfg["bands"]), n, n), np.float32)
    for i, band in enumerate(cfg["bands"]):
        if coadd.get("psf_model", "gaussian") != "moffat":
            images[i] = galsim.Gaussian(fwhm=coadd["fwhm_arcsec"][band]).drawImage(nx=n, ny=n, scale=pixscale).array
            continue
        size = 2 * coadd["psf_wings"]["half_pix"] + 1  # point sources are drawn on stamps up to this size
        core = moffat_core(coadd, band).drawImage(nx=size, ny=size, scale=pixscale).array.astype(float)
        full = fftconvolve(core, wings_kernel(band, coadd, cfg)[0], mode="same")
        start = size // 2 - n // 2
        images[i] = full[start:start + n, start:start + n]
    return images
