"""Stars for the mocks: catalogue rows with type "star", their rendering, and LSST-like saturated cores.

Stellar properties are calibrated on the stars of a real LSSTCam deep coadd; settings are in
config.STAR_CONFIG.
* Magnitudes follow the field's star counts, plus extra bright stars so that training sees spikes and saturation.
* Colours are resampled from field stars of similar magnitude (data/star_colours.csv), so red M-dwarf colours are
  common among faint stars and rare among bright ones.
* Light: a seeing core, a Moffat halo with a soft edge and radial streaks (~4-5% of the flux), and diffraction spikes
  whose brightness falls as ~t^-2 along them. All stars in an image share its spike angles (1-3 camera rotations
  per band, each giving two perpendicular lines); each arm's brightness varies.
* Saturated cores: each row of the saturated region holds one flat value, varying
  from row to row and band to band, which gives the coloured 'stripes' seen real bright stars.
"""

from pathlib import Path

import galsim
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, map_coordinates

from .config import STAR_CONFIG

STAR_COLOURS_PATH = Path(__file__).resolve().parent.parent / "data" / "star_colours.csv"
BAND_WAVELENGTH_NM = dict(u=367, g=483, r=622, i=755, z=868, y=971)
ZEROPOINT = 31.4


# --------------------------------------------------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------------------------------------------------

def draw_magnitudes(n, rng, mag_range, slope):
    """n r magnitudes with dN/dm proportional to 10^(slope m) between mag_range."""
    lo, hi = (10 ** (slope * m) for m in mag_range)
    return np.log10(lo + rng.uniform(size=n) * (hi - lo)) / slope


def colour_weights(mag_r, colours, star_cfg):
    """Probability of taking each field star's colours for a star of magnitude mag_r."""
    field_mags = colours["mag_r"].to_numpy()
    reference = max(mag_r, field_mags.min())
    weights = np.exp(-0.5 * ((field_mags - reference) / star_cfg["colour_match_mag"]) ** 2)
    if mag_r < field_mags.min():  # brighter than any field star: red colours fade out
        red = colours["g_r"].to_numpy() > star_cfg["red_g_r"]
        weights[red] *= 10 ** (-star_cfg["red_drop_per_mag"] * (field_mags.min() - mag_r))
    return weights / weights.sum()


def draw_stars(npix, pixscale, rng, star_cfg=STAR_CONFIG):
    """Stars at random positions in an npix x npix frame, with magnitudes, fluxes and per-star shape parameters."""
    area_deg2 = (npix * pixscale / 3600) ** 2
    mag_r = np.r_[draw_magnitudes(rng.poisson(star_cfg["density_per_deg2"] * area_deg2), rng, star_cfg["mag_range"],
                                  star_cfg["count_slope"]),
                  draw_magnitudes(rng.poisson(star_cfg["bright_per_deg2"] * area_deg2), rng,
                                  star_cfg["bright_mag_range"], star_cfg["count_slope"])]
    n = len(mag_r)
    colours = pd.read_csv(STAR_COLOURS_PATH)
    picked = colours.iloc[[rng.choice(len(colours), p=colour_weights(m, colours, star_cfg)) for m in mag_r]]
    jitter = lambda: rng.normal(0, 0.03, n)
    mags = dict(r=mag_r)
    mags["g"] = mag_r + picked["g_r"].to_numpy() + jitter()
    mags["u"] = mags["g"] + picked["u_g"].to_numpy() + jitter()
    mags["i"] = mag_r - picked["r_i"].to_numpy() + jitter()
    mags["z"] = mags["i"] - picked["i_z"].to_numpy() + jitter()
    mags["y"] = mags["z"] - picked["z_y"].to_numpy() + jitter()
    uniform = lambda key: rng.uniform(*star_cfg[key], n)
    stars = pd.DataFrame(dict(type="star", x_pix=rng.uniform(0, npix, n), y_pix=rng.uniform(0, npix, n)))
    for band in "ugrizy":
        stars[f"mag_{band}_total"] = mags[band]
        stars[f"flux_{band}_total"] = 10 ** (-0.4 * (mags[band] - ZEROPOINT))
    stars["spike_scale"] = np.exp(rng.normal(0, star_cfg["spike_scale_sigma"], n))
    for key in ("spike_core_arcsec", "spike_slope", "spike_width", "halo_cut_arcsec", "streak_strength"):
        stars[key] = uniform(key)
    stars["saturation_r"] = np.exp(rng.uniform(*np.log(star_cfg["saturation_r"]), n))
    return stars


def add_stars(galaxies, cfg, rng, star_cfg=STAR_CONFIG):
    """The catalogue with star rows (type "star") appended; galaxy rows are unchanged."""
    stars = draw_stars(cfg["npix"], cfg["pixscale"], rng, star_cfg)
    stars.insert(0, "id", np.arange(len(stars)) + (int(galaxies["id"].max()) + 1 if len(galaxies) else 0))
    print(f"Added {len(stars):,} stars (brightest r = {stars['mag_r_total'].min():.1f})")
    return pd.concat([galaxies, stars], ignore_index=True)


# --------------------------------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------------------------------

def spike_angles(rng, bands, star_cfg=STAR_CONFIG):
    """{band: [(angle_deg, weight), ...]} for one image: 1-3 random camera rotations per band, usually one dominant,
    each giving two perpendicular spike lines. Weights sum to 1 per band."""
    angles = {}
    for band in bands:
        rotations = rng.uniform(0, 90, rng.integers(star_cfg["spike_rotations"][0], star_cfg["spike_rotations"][1] + 1))
        weights = rng.dirichlet(0.5 * np.ones(len(rotations)))
        angles[band] = [(a, w / 2) for rotation, w in zip(rotations, weights) for a in (rotation, rotation + 90)]
    return angles


def pixel_grid(half):
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    return xx, yy


def stamp_half_size(mag_r):
    """Big enough for the halo and spikes of bright stars, small for faint ones."""
    return int(np.clip(40 + 30 * (18 - mag_r), 40, 300))


def radial_streaks(half, rng, strength, pixscale, start_arcsec=8.0, angular_scale_deg=0.6):
    """Multiplicative texture of fine radial streaks in the halo, growing from start_arcsec outwards."""
    xx, yy = pixel_grid(half)
    n = int(360 / (angular_scale_deg / 3))
    field = np.convolve(rng.normal(size=n + 60), np.exp(-0.5 * (np.arange(-30, 31) / 3) ** 2), mode="same")[30:-30]
    along = np.interp(np.degrees(np.arctan2(yy, xx)) % 360, np.linspace(0, 360, n, endpoint=False),
                      (field - field.mean()) / field.std(), period=360)
    grow = np.clip(np.hypot(xx, yy) * pixscale / start_arcsec - 1, 0, 1)
    return np.clip(1 + strength * along * grow, 0.05, None)


def spike_line(half, angle_deg, width_fwhm, core_arcsec, slope, arm_weights, pixscale, taper_arcsec):
    """One spike line: on-axis brightness 1 at 10" (mean of its two arms), falling as (1 + (t/core)^2)^(-slope/2),
    Gaussian across, each arm weighted separately, faded out inside taper_arcsec."""
    xx, yy = pixel_grid(half)
    a = np.radians(angle_deg)
    along = (xx * np.cos(a) + yy * np.sin(a)) * pixscale
    across = (-xx * np.sin(a) + yy * np.cos(a)) * pixscale
    line = ((1 + (along / core_arcsec) ** 2) / (1 + (10.0 / core_arcsec) ** 2)) ** (-slope / 2)
    line *= np.exp(-0.5 * (across / (width_fwhm / 2.355)) ** 2)
    line *= 1 - np.exp(-(np.hypot(xx, yy) * pixscale / taper_arcsec) ** 2)
    return 2 * line * np.where(along >= 0, arm_weights[0], arm_weights[1]) / sum(arm_weights)


def star_stamp(star, half, psf_fwhm, angles, cfg, rng, star_cfg=STAR_CONFIG):
    """(stamp, spikes): (band, 2h+1, 2h+1) images of one star in nJy/pixel, the whole star (core + halo + spikes, each
    band at its PSF) and its diffraction spikes alone."""
    size, pixscale = 2 * half + 1, cfg["pixscale"]
    streaks = radial_streaks(half, rng, star.streak_strength, pixscale)
    taper = 1 / (1 + (np.hypot(*pixel_grid(half)) * pixscale / star.halo_cut_arcsec) ** 6)
    halos = [galsim.Moffat(beta=beta, fwhm=fwhm).drawImage(nx=size, ny=size, scale=pixscale).array * taper * streaks
             for beta, fwhm in star_cfg["halo_basis"]]
    stamp, spike_stamp = np.zeros((len(cfg["bands"]), size, size)), np.zeros((len(cfg["bands"]), size, size))
    for i, band in enumerate(cfg["bands"]):
        flux = getattr(star, f"flux_{band}_total")
        core_arcsec = star.spike_core_arcsec * BAND_WAVELENGTH_NM[band] / BAND_WAVELENGTH_NM["r"]  # longer when redder
        spikes = sum(weight * spike_line(half, angle, star.spike_width * psf_fwhm[band], core_arcsec, star.spike_slope,
                                         np.exp(rng.normal(0, star_cfg["arm_scatter"], 2)), pixscale,
                                         star_cfg["spike_taper_arcsec"])
                     for angle, weight in angles[band])
        spikes = flux * star_cfg["spike_at_10arcsec"][band] * star.spike_scale * spikes
        halo = flux * sum(f * h for f, h in zip(star_cfg["halo_fraction"][band], halos))
        core_fraction = 1 - sum(star_cfg["halo_fraction"][band]) - spikes.sum() / flux
        core = galsim.Gaussian(fwhm=psf_fwhm[band]).drawImage(nx=size, ny=size, scale=pixscale).array
        stamp[i] = flux * core_fraction * core + halo + spikes
        spike_stamp[i] = spikes
    return stamp, spike_stamp


def add_stamp(image, stamp, x, y):
    """Add a (band, 2h+1, 2h+1) stamp centred on integer pixel (x, y) of a (band, ny, nx) image, clipped at edges."""
    h = stamp.shape[1] // 2
    ny, nx = image.shape[1:]
    y0, y1, x0, x1 = max(y - h, 0), min(y + h + 1, ny), max(x - h, 0), min(x + h + 1, nx)
    if y1 > y0 and x1 > x0:
        image[:, y0:y1, x0:x1] += stamp[:, y0 - (y - h):y1 - (y - h), x0 - (x - h):x1 - (x - h)]


def render_stars(stars, shape, origin, psf_fwhm, cfg, rng, star_cfg=STAR_CONFIG, spikes_out=None):
    """(band, ny, nx) image of the catalogue's stars, on the canvas whose frame pixel origin is at image[:, 0, 0].

    If spikes_out (an array of the same shape) is given, the diffraction spikes alone are also added to it: the truth
    a detector can learn spike pixels from."""
    image = np.zeros(shape, np.float32)
    angles = spike_angles(rng, cfg["bands"], star_cfg)
    for star in stars.itertuples():
        stamp, spikes = star_stamp(star, stamp_half_size(star.mag_r_total), psf_fwhm, angles, cfg, rng, star_cfg)
        x, y = int(round(star.x_pix)) - origin[0], int(round(star.y_pix)) - origin[1]
        add_stamp(image, stamp, x, y)
        if spikes_out is not None:
            add_stamp(spikes_out, spikes, x, y)
    return image


# --------------------------------------------------------------------------------------------------------------------
# Saturated cores (applied to each coadd after its noise)
# --------------------------------------------------------------------------------------------------------------------

def smooth_series(n, rng, correlation):
    """n values of smooth, unit-variance random noise (Gaussian-smoothed white noise)."""
    kernel = np.exp(-0.5 * (np.arange(-4 * correlation, 4 * correlation + 1) / correlation) ** 2)
    series = np.convolve(rng.normal(size=n + len(kernel)), kernel, mode="same")[len(kernel) // 2:][:n]
    return (series - series.mean()) / max(series.std(), 1e-9)


def fill_saturated_core(image, clean, level, rng, stretch, shift, star_cfg=STAR_CONFIG, mask_out=None):
    """Replace a star's saturated core in `image` (one band, stamp-sized) with flat rows, as LSST coadds show it.

    The region is where the noise-free `clean` image exceeds `level`, stretched vertically and shifted (shared by all
    bands of a star). Each row holds level x exp(row_scatter x smooth noise over rows); a 2-pixel rim is dimmed.
    If mask_out (a boolean array shaped like image) is given, every pixel replaced or dimmed here is set True in it.
    """
    ny, nx = clean.shape
    yc, xc = np.unravel_index(np.argmax(clean), clean.shape)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    saturated = map_coordinates(clean, [yc + (yy - yc - shift[0]) / stretch, xx - shift[1]], order=1) > level
    rows = np.flatnonzero(saturated.any(axis=1))
    if len(rows) == 0:
        return image
    values = level * np.exp(star_cfg["row_scatter"] * smooth_series(len(rows), rng, star_cfg["row_correlation"]))
    jitter = [np.rint(0.8 * smooth_series(len(rows), rng, 1.0)).astype(int) for _ in range(2)]
    filled = np.zeros_like(saturated)
    for row, value, left, right in zip(rows, values, *jitter):
        cols = np.flatnonzero(saturated[row])
        start, stop = max(cols[0] - left, 0), min(cols[-1] + right, nx - 1)
        filled[row, start:stop + 1] = True
        image[row, start:stop + 1] = value
    rim = binary_dilation(filled, iterations=2) & ~filled
    image[rim] *= 1 - 0.15 * rng.uniform()
    if mask_out is not None:
        mask_out |= filled | rim
    return image


def saturate_stars(signal, clean, stars, origin, bands, rng, star_cfg=STAR_CONFIG, mask_out=None):
    """Apply saturated cores to a noised coadd `signal`, given its noise-free version `clean` (both band, ny, nx).

    If mask_out (a boolean array shaped like signal) is given, the pixels of every saturated core are set True in it:
    the coadd's SAT mask plane. The random draws are the same with or without it."""
    ny, nx = signal.shape[1:]
    for star in stars.itertuples():
        x, y = int(round(star.x_pix)) - origin[0], int(round(star.y_pix)) - origin[1]
        h = min(stamp_half_size(star.mag_r_total), 60)  # saturated cores are at most a few arcsec across
        if not (h <= x < nx - h and h <= y < ny - h):
            continue
        box = (slice(y - h, y + h + 1), slice(x - h, x + h + 1))
        stretch, shift = rng.uniform(*star_cfg["core_stretch"]), rng.normal(0, star_cfg["core_shift_px"], 2)
        for i, band in enumerate(bands):
            level = star.saturation_r * star_cfg["saturation_ratio"][band]
            if clean[i][box].max() > level:
                band_mask = None if mask_out is None else mask_out[i][box]  # a view: filled in place
                signal[i][box] = fill_saturated_core(signal[i][box].copy(), clean[i][box], level, rng, stretch, shift,
                                                     star_cfg, band_mask)
    return signal
