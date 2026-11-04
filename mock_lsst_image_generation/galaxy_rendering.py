"""Render a mock catalogue (galaxies, star-forming regions, tidal blobs) into a noise-free ugrizy image cube with
GalSim.

Bulges and discs are untruncated Sersic profiles (n = 4 / 1 when the catalogue has none) sharing the galaxy's position
angle; star-forming regions and tidal blobs are Gaussians. Each object is convolved with its band's PSF (a Gaussian
FWHM or a GalSim profile; psf.py) and drawn on its own stamp, sized by GalSim so that no flux is clipped, then added to
the canvas. Positions use GalSim's 1-indexed pixel convention: canvas pixel (x, y) is image[:, y - 1, x - 1], and frame
pixel x_pix sits at canvas x = x_pix - x_min + 1."""

import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import galsim
import numpy as np
from tqdm import tqdm

from .bcg_sources import render_bcg
from .galaxy_structure import add_structure, has_structure
from .psf import psf_profile
from .stars import render_stars, stamp_half_size

# GalSim >= 2.8 refuses very large FFTs unless this is False; big, extended galaxies must still be drawn in full.
if hasattr(galsim.errors, "raise_fft_size_error"):
    galsim.errors.raise_fft_size_error = False

COMPONENTS = [("bulge", 4.0), ("disc", 1.0)]  # (name, default Sersic index)
STRUCTURE_COLUMNS = ["hubble_type", "n_arms", "arm_pitch_deg", "arm_strength", "arm_sharpness", "arm_phase_deg",
                     "arm_start_h", "barred", "irregularity", "structure_seed"]  # from hubble_types.add_hubble_types


def positive_flux(record, column):
    """The record's flux in column, or 0 if it is missing, non-finite or non-positive."""
    flux = float(record.get(column, 0.0))
    return flux if np.isfinite(flux) and flux > 0 else 0.0


def draw_stamp(band_profiles, support_profiles, x, y, n_bands, pixscale, min_stamp_pix):
    """Draw {band index: profile} on one odd-sized square stamp centred on (x, y), big enough for every profile in
    support_profiles. Returns (xmin, ymin, cube) with cube of shape (n_bands, size, size)."""
    size = max(int(np.ceil(max(profile.getGoodImageSize(pixscale) for profile in support_profiles))), min_stamp_pix)
    size += 1 - size % 2
    half = size // 2
    ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
    bounds = galsim.BoundsI(ix - half, ix + half, iy - half, iy + half)
    centre = galsim.PositionD(x, y)
    cube = np.zeros((n_bands, size, size), dtype=np.float32)
    for band_index, profile in band_profiles.items():
        stamp = galsim.ImageF(bounds, scale=pixscale)
        profile.drawImage(image=stamp, center=centre)
        cube[band_index] = stamp.array
    return bounds.xmin, bounds.ymin, cube


def render_galaxy(record, psf, bands, pixscale, min_stamp_pix, structure_min_re_arcsec=None):
    """Bulge + disc stamp for one galaxy record, or None if neither component has a valid size, shape and flux.
    Discs with re_disc_arcsec >= structure_min_re_arcsec also get the arms / bar / clumps of their Hubble type."""
    if record.get("profile") == "core_sersic":
        return render_bcg(record, psf, bands, pixscale)
    angle = float(record["pa_deg"]) * galsim.degrees
    components = {}
    for name, default_n in COMPONENTS:
        re, ellipticity = float(record[f"re_{name}_arcsec"]), float(record[f"ellipticity_{name}"])
        has_flux = any(positive_flux(record, f"flux_{band}_{name}") > 0 for band in bands)
        if not (np.isfinite(re) and re > 0 and np.isfinite(ellipticity) and has_flux):
            continue
        n = float(record[f"n_{name}"]) if np.isfinite(record[f"n_{name}"]) else default_n
        axis_ratio = float(np.clip(1.0 - ellipticity, 0.05, 1.0))
        components[name] = galsim.Sersic(n=n, half_light_radius=re, flux=1.0).shear(q=axis_ratio, beta=angle)
    if not components:
        return None

    # The stamp must hold every component on its own, so a faint extended disc is not cut off by a compact bulge.
    band_profiles, support_profiles = {}, []
    for band_index, band in enumerate(bands):
        band_psf = psf_profile(psf[band])
        parts = [component.withFlux(flux) for name, component in components.items()
                 if (flux := positive_flux(record, f"flux_{band}_{name}")) > 0]
        support_profiles += [galsim.Convolve([part, band_psf]) for part in parts]
        if parts:
            band_profiles[band_index] = galsim.Convolve([parts[0] if len(parts) == 1 else galsim.Add(parts), band_psf])
    xmin, ymin, cube = draw_stamp(band_profiles, support_profiles, float(record["x_img"]), float(record["y_img"]),
                                  len(bands), pixscale, min_stamp_pix)
    if structure_min_re_arcsec is not None and "disc" in components and has_structure(record, structure_min_re_arcsec):
        add_structure(cube, xmin, ymin, record, bands, psf, pixscale)
    return xmin, ymin, cube


def render_blob(record, psf, bands, pixscale, min_stamp_pix):
    """Gaussian stamp for one star-forming region or tidal blob record, or None if it has no valid position, width or
flux."""
    x, y, sigma = float(record["x_img"]), float(record["y_img"]), float(record["sigma_arcsec"])
    fluxes = [positive_flux(record, f"flux_{band}") for band in bands]
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(sigma) and sigma > 0 and any(fluxes)):
        return None
    blob = galsim.Gaussian(sigma=sigma, flux=1.0)
    band_profiles = {i: galsim.Convolve([blob.withFlux(flux), psf_profile(psf[band])])
                     for i, (band, flux) in enumerate(zip(bands, fluxes)) if flux > 0}
    return draw_stamp(band_profiles, band_profiles.values(), x, y, len(bands), pixscale, min_stamp_pix)


def render_point_source(record, psf, bands, pixscale, min_stamp_pix):
    """Stamp of one star drawn as its band's PSF profile, on a stamp as large as stars.py gives the star, or None if
    it has no flux."""
    band_profiles = {i: psf_profile(psf[band]).withFlux(flux) for i, band in enumerate(bands)
                     if (flux := positive_flux(record, f"flux_{band}_total")) > 0}
    if not band_profiles:
        return None
    size = max(min_stamp_pix, 2 * stamp_half_size(record["mag_r_total"]) + 1)
    return draw_stamp(band_profiles, band_profiles.values(), record["x_img"], record["y_img"], len(bands), pixscale,
                      size)


def add_stamp(image, xmin, ymin, cube):
    """Add a stamp to the image in place, clipping it at the image edges."""
    ny, nx = image.shape[1:]
    x0, x1 = max(xmin, 1), min(xmin + cube.shape[2] - 1, nx)
    y0, y1 = max(ymin, 1), min(ymin + cube.shape[1] - 1, ny)
    if x0 <= x1 and y0 <= y1:
        image[:, y0 - 1:y1, x0 - 1:x1] += cube[:, y0 - ymin:y1 - ymin + 1, x0 - xmin:x1 - xmin + 1]


def render_records(render, records, shape, psf, cfg, n_workers, label):
    """Render every record onto a new zero image of the given shape, in parallel over n_workers processes.

    Stamps are added in record order whatever order the workers finish in, so the image is reproducible exactly.
    """
    image = np.zeros(shape, dtype=np.float32)
    args = (psf, cfg["bands"], cfg["pixscale"], cfg["min_stamp_pix"])
    with tqdm(total=len(records), desc=f"Rendering {label}") as progress:
        def add(stamp):
            if stamp is not None:
                add_stamp(image, *stamp)
            progress.update()

        if n_workers <= 1:
            for record in records:
                add(render(record, *args))
            return image
        with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("fork")) as pool:
            pending = deque()
            for record in records:
                pending.append(pool.submit(render, record, *args))
                if len(pending) >= 4 * n_workers:  # bounds memory: large stamps can be hundreds of MB
                    add(pending.popleft().result())
            while pending:
                add(pending.popleft().result())
    return image


def sfregion_blobs(sfregions, bands):
    """Star-forming region table in the blob form render_blob reads: x_pix, y_pix, sigma_arcsec, flux_<band>."""
    return sfregions.rename(columns={"x_pix_sfregion": "x_pix", "y_pix_sfregion": "y_pix",
                                  **{f"flux_{band}_sfregion": f"flux_{band}" for band in bands}})


def tidal_blobs(tidal, bands, pixscale):
    """Tidal table (one row per blob and band) in blob form, one row per blob with a flux_<band> column each."""
    tidal = tidal.assign(sigma_arcsec=tidal["sigma_pix"] * pixscale)
    wide = tidal.pivot_table(index=["tidal_pair_id", "tidal_blob_id", "x_pix_tidal", "y_pix_tidal", "sigma_arcsec"],
                             columns="band", values="flux_tidal", aggfunc="sum", fill_value=0.0).reset_index()
    wide = wide.rename(columns={"x_pix_tidal": "x_pix", "y_pix_tidal": "y_pix",
                                **{band: f"flux_{band}" for band in bands}})
    for band in bands:
        if f"flux_{band}" not in wide.columns:
            wide[f"flux_{band}"] = 0.0
    return wide


def to_records(table, columns, required, origin):
    """Rows as dicts of the given columns, with canvas coordinates x_img / y_img; rows missing a required value are
    dropped."""
    x_min, y_min = origin
    table = table.assign(x_img=table["x_pix"] - x_min + 1, y_img=table["y_pix"] - y_min + 1)
    return table.dropna(subset=required)[columns].to_dict("records")


def split_stars(table):
    """(galaxy rows, star rows) of a catalogue table."""
    is_star = (table["type"] == "star").to_numpy() if "type" in table else np.zeros(len(table), bool)
    return table[~is_star], table[is_star]


# Light rendered into the image that render_catalogue can also return on its own (truth for learning where it is).
COMPONENT_IMAGES = ("sfregions", "tidal", "spikes")


def render_catalogue(catalogue, psf, cfg, n_workers=1, star_seed=0, components_out=None):
    """Noise-free (band, y, x) float32 image of a catalogue's galaxies, star-forming regions, tidal blobs and stars (nJy
    per pixel), with each band convolved with psf[band]: a Gaussian FWHM (arcsec), or a GalSim profile.

    The canvas spans the galaxy centres, or frame pixels [0, cfg['canvas_npix']) on both axes if that is set; returns
    (image, origin) where origin = (x_min, y_min) is the frame pixel at image[:, 0, 0]. With Gaussian FWHMs, stars
    add their own halo and diffraction spikes (stars.py) and star_seed seeds their random details (spike angles, arm
    brightness, halo streaks); with GalSim profiles they are point sources, and the halo and spikes are left to the
    coadd's PSF (psf.py). If components_out is a dict, it receives an image of the same shape for each of
    COMPONENT_IMAGES: the light of the star-forming regions, of the tidal features and of the stars' diffraction spikes
    alone (zero for point sources).
    """
    bands, (galaxies, stars) = cfg["bands"], split_stars(catalogue.galaxies)
    if cfg.get("canvas_npix"):
        npix = int(cfg["canvas_npix"])
        origin, shape = (0, 0), (len(bands), npix, npix)
    else:
        x_min, x_max = int(np.floor(galaxies["x_pix"].min())), int(np.ceil(galaxies["x_pix"].max()))
        y_min, y_max = int(np.floor(galaxies["y_pix"].min())), int(np.ceil(galaxies["y_pix"].max()))
        origin, shape = (x_min, y_min), (len(bands), y_max - y_min + 1, x_max - x_min + 1)

    # Disc columns are NaN for bulge-only galaxies; render_galaxy handles that, so only these must be present.
    galaxy_columns = ["x_img", "y_img", "pa_deg", "re_bulge_arcsec", "re_disc_arcsec", "ellipticity_bulge",
                      "ellipticity_disc", "n_bulge", "n_disc",
                      *[f"flux_{band}_{name}" for band in bands for name, _ in COMPONENTS]]
    galaxy_columns += [column for column in STRUCTURE_COLUMNS if column in galaxies]
    galaxy_columns += [column for column in ["profile", "re_total_arcsec", *[f"flux_{band}_total" for band in bands]]
                       if column in galaxies]
    galaxy_records = to_records(galaxies, galaxy_columns, ["x_img", "y_img", "pa_deg"], origin)
    render = partial(render_galaxy, structure_min_re_arcsec=cfg.get("structure_min_re_arcsec"))
    image = render_records(render, galaxy_records, shape, psf, cfg, n_workers, "galaxies")

    components: dict[str, np.ndarray] = {name: np.zeros(shape, np.float32) for name in COMPONENT_IMAGES}
    blob_columns = ["x_img", "y_img", "sigma_arcsec", *[f"flux_{band}" for band in bands]]
    blob_tables = [("sfregions", catalogue.sfregions, lambda table: sfregion_blobs(table, bands)),
                   ("tidal", catalogue.tidal_blobs, lambda table: tidal_blobs(table, bands, cfg["pixscale"]))]
    for name, table, to_blobs in blob_tables:
        if len(table):
            blob_records = to_records(to_blobs(table), blob_columns, ["x_img", "y_img", "sigma_arcsec"], origin)
            components[name] = render_records(render_blob, blob_records, shape, psf, cfg, n_workers,
                                              "star-forming regions" if name == "sfregions" else "tidal blobs")
            image += components[name]
    if len(stars) and all(isinstance(band_psf, galsim.GSObject) for band_psf in psf.values()):
        star_columns = ["x_img", "y_img", "mag_r_total", *[f"flux_{band}_total" for band in bands]]
        star_records = to_records(stars, star_columns, ["x_img", "y_img"], origin)
        image += render_records(render_point_source, star_records, shape, psf, cfg, n_workers, "stars")
    elif len(stars):
        image += render_stars(stars, shape, origin, psf, cfg, np.random.default_rng([star_seed, 2]),
                              spikes_out=components["spikes"])
    if components_out is not None:
        components_out.update(components)
    return image, origin
