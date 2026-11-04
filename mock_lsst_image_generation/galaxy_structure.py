"""Spiral arms, bars and irregular clumps for resolved discs, as light moved around within the catalogue's disc.

The disc's own light is redistributed by azimuthal Fourier modes in the disc plane (r, theta), so band fluxes are
unchanged:
    arms         1 + A [w(m (theta - ln(r / r0) / tan(pitch))) - 1], w = ((1 + cos) / 2)^k normalised to average 1
                 around each ring: m logarithmic spiral arms starting at r0
    bar          inside r0 the disc's light is flattened and gathered into an m = 2 mode lined up with where the arms
                 start, so the arms grow from the bar's ends
    irregular    random low-order modes with a radial twist (lopsided, clumpy light)
The disc is 3D (Gaussian vertical profile, scale height DISC_THICKNESS x the scale length) and seen at the inclination
its catalogue axis ratio implies, via the exact line-of-sight projection
    (1 / cos i) integral of Sigma(x, (y + z sin i) / cos i) f(z) dz          (x along the major axis)
done by Gauss-Hermite quadrature. The result is two images per unit disc flux, each summing to zero: the bar part
(the disc's colours) and the arm / clump part (bluer: scaled per band by ARM_COLOUR_BOOST). GalSim's smooth disc
stays as it is; these images are added on top of its stamp, convolved with the same PSF.
"""

import numpy as np
from scipy.special import gammaincinv, gammaln

from .psf import convolve_psf

ARM_COLOUR_BOOST = dict(u=1.4, g=1.25, r=1.0, i=0.8, z=0.7, y=0.65)  # arm contrast per band (young stars are blue)
BAR_STRENGTH, BAR_SHARPNESS, BAR_FLATNESS = 0.9, 6.0, 4.0  # bar mode amplitude, power, flattening of the disc inside
DISC_THICKNESS = 0.12  # vertical scale height / radial scale length
MIN_COS_INCLINATION = 0.05  # as galaxy_rendering clips the disc axis ratio
N_VERTICAL_NODES = 3
N_IRREGULAR_MODES = 5
EXTENT_H = 6.0  # structure is computed out to this many scale lengths from the centre
OVERSAMPLE = 3  # sub-pixels per pixel side: pixel-integrated like GalSim's disc, so thin discs subtract cleanly
STRIP_CELLS = 4_000_000  # grid cells worked on at once by structure_images (each float64 array is then 32 MB)
SERSIC_B1 = 1.678  # half-light radius / scale length of an exponential disc


def mode_mean(k):
    """Mean of ((1 + cos) / 2)^k over a full turn."""
    return np.exp(gammaln(k + 0.5) - gammaln(k + 1)) / np.sqrt(np.pi)


def face_on_disc(u, v, record, h, modes):
    """(smooth disc, disc with its bar, the same with its arms / clumps) surface brightness at disc-plane coordinates
    (u, v)."""
    n = float(record["n_disc"]) if np.isfinite(record["n_disc"]) else 1.0
    re = float(record["re_disc_arcsec"])
    r, theta = np.hypot(u, v), np.arctan2(v, u)
    smooth = np.exp(-gammaincinv(2 * n, 0.5) * (r / re) ** (1 / n))

    r0, phase = record["arm_start_h"] * h, np.radians(record["arm_phase_deg"])
    start = 1 / (1 + np.exp(-(r - r0) / (0.25 * h)))
    barred = smooth
    if record["barred"]:
        bar_mode = ((1 + np.cos(2 * (theta + phase))) / 2) ** BAR_SHARPNESS / mode_mean(BAR_SHARPNESS) - 1
        centre = 1 - np.exp(-(r / (0.12 * r0)) ** 2)
        flatten = np.where(r < r0, np.exp((r - r0) * (1 - 1 / BAR_FLATNESS) / h), 1)
        barred = smooth * flatten * (1 + BAR_STRENGTH * (1 - start) * centre * bar_mode)

    pattern = np.zeros_like(r)
    if record["n_arms"] > 0:
        k = record["arm_sharpness"]
        psi = record["n_arms"] * (theta - np.log(np.maximum(r, 1e-3) / r0) / np.tan(np.radians(record["arm_pitch_deg"]))
                                  + phase)
        pattern += record["arm_strength"] * start * (((1 + np.cos(psi)) / 2) ** k / mode_mean(k) - 1)
    for amplitude, mode_phase, twist, m in modes:
        pattern += amplitude * np.cos(m * theta + mode_phase + twist * r / h)
    return smooth, barred, barred * np.maximum(1 + pattern, 0.02)


def irregular_modes(record):
    """(amplitude, phase, radial twist, m) of an irregular's random low-order modes, fixed by its structure_seed."""
    if not record["irregularity"] > 0:
        return []
    rng = np.random.default_rng(int(record["structure_seed"]))
    return [(record["irregularity"] / m ** 0.7, rng.uniform(0, 2 * np.pi), rng.normal(0, 1.5), m)
            for m in range(1, N_IRREGULAR_MODES + 1)]


def structure_weight(record, u, v):
    """Surface brightness of the disc with its arms / bar / clumps relative to the smooth disc, at disc-plane
    coordinates (u, v) in the midplane: how much more light (so how many more stars) the structure puts there."""
    h = float(record["re_disc_arcsec"]) / SERSIC_B1
    smooth, _, shaped = face_on_disc(u, v, record, h, irregular_modes(record))
    return shaped / np.maximum(smooth, 1e-300)


def disc_layers(record, x_offsets, y_offsets):
    """(smooth, barred, shaped) disc surface brightness, unnormalised, on a grid of sky offsets (arcsec) from the
    galaxy centre, integrated through the disc's thickness."""
    h = float(record["re_disc_arcsec"]) / SERSIC_B1
    pa = np.radians(float(record["pa_deg"]))
    x, y = np.meshgrid(x_offsets, y_offsets)
    major, minor = x * np.cos(pa) + y * np.sin(pa), -x * np.sin(pa) + y * np.cos(pa)
    cos_i = min(max(1.0 - float(record["ellipticity_disc"]), MIN_COS_INCLINATION), 1.0)
    sin_i = np.sqrt(1 - cos_i ** 2)

    modes = irregular_modes(record)
    nodes, weights = np.polynomial.hermite_e.hermegauss(N_VERTICAL_NODES)
    smooth, barred, shaped = np.zeros_like(major), np.zeros_like(major), np.zeros_like(major)
    for z, weight in zip(nodes * DISC_THICKNESS * h, weights / weights.sum()):
        # GalSim's shear keeps area (re is the circularised radius), so the disc plane is scaled by sqrt(cos i) too.
        layer = face_on_disc(major * np.sqrt(cos_i), (minor + z * sin_i) / np.sqrt(cos_i), record, h, modes)
        smooth, barred, shaped = smooth + weight * layer[0], barred + weight * layer[1], shaped + weight * layer[2]
    return smooth, barred, shaped


def structure_images(record, x_offsets, y_offsets, bin_size=1):
    """(bar, arms, depth, disc) for a grid of sky offsets (arcsec) from the galaxy centre: the bar and arm / clump
    images per unit disc flux, the deepest fraction of the barred disc's light the arms remove anywhere, and the
    barred disc itself (unit sum). Images are summed over bin_size x bin_size blocks of the grid.

    A large, inclined disc's grid can be far too big for memory, so it is worked through in strips of rows: a first
    pass finds the totals the images are normalised by, a second builds and bins each strip."""
    rows = max(1, STRIP_CELLS // (len(x_offsets) * bin_size)) * bin_size
    strips = [slice(start, start + rows) for start in range(0, len(y_offsets), rows)]
    layers = disc_layers(record, x_offsets, y_offsets[strips[0]]) if len(strips) == 1 else None
    smooth_sum = barred_sum = shaped_sum = barred_max = 0.0
    for strip in strips:
        smooth, barred, shaped = layers or disc_layers(record, x_offsets, y_offsets[strip])
        smooth_sum, barred_sum, shaped_sum = smooth_sum + smooth.sum(), barred_sum + barred.sum(), shaped_sum + shaped.sum()
        barred_max = max(barred_max, barred.max())

    binned = lambda a: a.reshape(a.shape[0] // bin_size, bin_size, a.shape[1] // bin_size, bin_size).sum(axis=(1, 3))
    bar, arms, disc, depth = [], [], [], -np.inf
    for strip in strips:
        smooth, barred, shaped = layers or disc_layers(record, x_offsets, y_offsets[strip])
        strip_arms = shaped * barred_sum / shaped_sum - barred  # rescaled so arms / clumps only move light
        depth = max(depth, float(np.max(-strip_arms / np.maximum(barred, 1e-30 * barred_max))))
        bar.append(binned(barred / barred_sum - smooth / smooth_sum))
        arms.append(binned(strip_arms / barred_sum))
        disc.append(binned(barred / barred_sum))
    return np.vstack(bar), np.vstack(arms), depth, np.vstack(disc)


def has_structure(record, min_re_arcsec):
    """Whether a galaxy record has arms, a bar or clumps, on a disc large enough to show them."""
    if "hubble_type" not in record or not isinstance(record["hubble_type"], str):
        return False
    shaped = record["n_arms"] > 0 or record["barred"] or record["irregularity"] > 0
    return bool(shaped and np.isfinite(record["re_disc_arcsec"]) and record["re_disc_arcsec"] >= min_re_arcsec
                and np.isfinite(record["ellipticity_disc"]))


def add_structure(cube, xmin, ymin, record, bands, psf, pixscale):
    """Add a galaxy's arms / bar / clumps to its rendered stamp cube (band, y, x) in place, band by band scaled by
    its disc flux and convolved with the band's PSF (psf.convolve_psf). Only the part of the stamp within EXTENT_H
    disc scale lengths of the centre is touched."""
    size = cube.shape[1]
    q = min(max(1.0 - float(record["ellipticity_disc"]), MIN_COS_INCLINATION), 1.0)
    # GalSim stretches an inclined disc's major axis by 1 / sqrt(q), so the window must reach that far.
    half = int(np.ceil(EXTENT_H * float(record["re_disc_arcsec"]) / SERSIC_B1 / np.sqrt(q) / pixscale))
    ix, iy = int(round(record["x_img"])) - xmin, int(round(record["y_img"])) - ymin
    x0, x1, y0, y1 = max(ix - half, 0), min(ix + half + 1, size), max(iy - half, 0), min(iy + half + 1, size)
    if x0 >= x1 or y0 >= y1:
        return
    sub = (np.arange(OVERSAMPLE) + 0.5) / OVERSAMPLE - 0.5
    offsets = lambda start, stop, centre: ((np.arange(start, stop)[:, None] + sub).ravel() - centre) * pixscale
    bar, arms, depth, disc = structure_images(record, offsets(xmin + x0, xmin + x1, record["x_img"]),
                                              offsets(ymin + y0, ymin + y1, record["y_img"]), OVERSAMPLE)
    # Arms are bluer (ARM_COLOUR_BOOST) than the rest of the disc. The extra (or missing) light of each band's arms is
    # balanced across the whole disc in proportion to its light, not taken from between the arms, so the inter-arm
    # disc keeps the disc's colour instead of turning into a separate red glow.
    arm_light = np.maximum(arms, 0.0)
    boost_room = 0.9 * (1 - depth) / max(arm_light.sum(), 1e-3)  # the removal must leave inter-arm light positive
    for i, band in enumerate(bands):
        flux = float(record.get(f"flux_{band}_disc", 0.0))
        if not (np.isfinite(flux) and flux > 0):
            continue
        extra = min(ARM_COLOUR_BOOST.get(band, 1.0) - 1.0, boost_room)
        image = flux * (bar + arms + extra * (arm_light - arm_light.sum() * disc))
        cube[i, y0:y1, x0:x1] += convolve_psf(image, psf[band], pixscale).astype(cube.dtype)
