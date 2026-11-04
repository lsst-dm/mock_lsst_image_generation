"""Truth table of every peak-like component of a mock, and which of them are peaks in a coadd's noise-free image.

A detector that seeds a deblender (e.g. scarlet lite) should find one peak per component that shows up as a local
maximum: galaxy centres (including BCG and LSB galaxies), stars, star-forming regions and tidal-bridge blobs. Spiral
arms, bars and irregular modes are not listed: they move a disc's own light around instead of adding sources
(galaxy_structure.py), so a peak they make is matched to its galaxy or to a star-forming region on it.

truth_table lists every component with its frame position and per-band flux. find_truth_peaks takes the saved base
render to a coadd's PSF band by band (coadd_synthesis.coadd_psf_band), finds the local maxima of that noise-free image
whose per-pixel S/N (against the coadd's variance) is at least min_snr, and assigns each maximum to the nearest
component within match_radius pixels (ties go to the earlier row: galaxies, then stars, star-forming regions and
tidal blobs, since the end blobs of a tidal bridge sit on the galaxy centres). Each component then records, per band,
whether it owns a maximum, the S/N of its brightest one and that maximum's distance. Maxima with no component within
match_radius are kept in the peak table with truth_id -1.

Fluxes are in nJy. A galaxy's flux_<band> is its bulge + disc light only: the light of its star-forming regions and
tidal blobs is in their own rows.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from .coadd_synthesis import coadd_psf_band, sky_sigma
from .galaxy_rendering import sfregion_blobs, split_stars, tidal_blobs

KINDS = ("galaxy", "star", "sfregion", "tidal")
MAX_CANDIDATES = 8  # components considered per maximum when breaking distance ties


def column(table, name, default):
    """A column with missing values replaced by default, or default everywhere if the table lacks it."""
    if name in table:
        return table[name].where(table[name].notna(), default).to_numpy()
    return np.full(len(table), default, dtype=object if isinstance(default, str) else float)


def component_rows(kind, table, bands, flux_column, catalogue_id, parent_id, component_index, population,
                   hubble_type, tidal_pair_id):
    """Truth rows of one kind of component, in a common layout."""
    rows = pd.DataFrame(dict(kind=kind, catalogue_id=np.asarray(catalogue_id, int),
                             parent_id=np.asarray(parent_id, int), component_index=np.asarray(component_index, int),
                             tidal_pair_id=np.asarray(tidal_pair_id, int), population=population,
                             hubble_type=hubble_type, x_pix=table["x_pix"].to_numpy(float),
                             y_pix=table["y_pix"].to_numpy(float)))
    for band in bands:
        rows[f"flux_{band}"] = table[flux_column.format(band=band)].to_numpy(float)
    return rows


def truth_table(catalogue, bands, pixscale):
    """One row per peak-like component of a MockCatalogue (see the module docstring), with columns
    truth_id, kind (galaxy, star, sfregion, tidal), catalogue_id (galaxy / star id, else -1), parent_id (host galaxy
    of a star-forming region, donor galaxy of a tidal blob, else -1), component_index (star-forming region or tidal
    blob number within its parent, else -1), tidal_pair_id, population (lsb_population: extended_dirr, udg,
    almost_dark, bcg, or ''), hubble_type (the galaxy's, or the host's for a star-forming region), x_pix, y_pix and
    flux_<band>."""
    galaxies, stars = split_stars(catalogue.galaxies)
    none = lambda table: np.full(len(table), -1)
    hubble_of = pd.Series(column(galaxies, "hubble_type", ""), index=galaxies["id"].to_numpy(int))
    population_of = pd.Series(column(galaxies, "lsb_population", ""), index=galaxies["id"].to_numpy(int))
    blocks = [
        component_rows("galaxy", galaxies, bands, "flux_{band}_total", galaxies["id"], none(galaxies), none(galaxies),
                       column(galaxies, "lsb_population", ""), column(galaxies, "hubble_type", ""), none(galaxies)),
        component_rows("star", stars, bands, "flux_{band}_total", stars["id"], none(stars), none(stars), "", "",
                       none(stars)),
    ]
    if len(catalogue.sfregions):
        sfr = sfregion_blobs(catalogue.sfregions, bands)
        parents = sfr["parent_id"].to_numpy(int)
        blocks.append(component_rows("sfregion", sfr, bands, "flux_{band}", none(sfr), parents, sfr["sfregion_id"],
                                     population_of.reindex(parents).fillna("").to_numpy(),
                                     hubble_of.reindex(parents).fillna("").to_numpy(), none(sfr)))
    if len(catalogue.tidal_blobs):
        tidal = tidal_blobs(catalogue.tidal_blobs, bands, pixscale)
        donors = catalogue.tidal_blobs.groupby("tidal_pair_id")["donor_id"].first()
        parents = donors.reindex(tidal["tidal_pair_id"]).to_numpy(int)
        blocks.append(component_rows("tidal", tidal, bands, "flux_{band}", none(tidal), parents,
                                     tidal["tidal_blob_id"], population_of.reindex(parents).fillna("").to_numpy(),
                                     hubble_of.reindex(parents).fillna("").to_numpy(), tidal["tidal_pair_id"]))
    truth = pd.concat([block for block in blocks if len(block)], ignore_index=True)
    truth.insert(0, "truth_id", np.arange(len(truth)))
    return truth


def find_truth_peaks(truth, base_image, origin, coadd, cfg, min_snr=1.0, match_radius=3.0):
    """(truth, peaks) for one coadd (a coadd_manifest.json / coadd_grid entry: its fwhm_arcsec and n_visit per band).

    truth gains x_array / y_array (array pixel coordinates: frame position minus origin), in_image, and per band
    peak_<band> (owns a local maximum), n_peaks_<band>, peak_snr_<band> and peak_dist_<band> (S/N of its brightest
    maximum and that maximum's distance in pixels; NaN if none), plus n_peak_bands. peaks holds every local maximum of
    the noise-free image with S/N >= min_snr: band, x_array, y_array, x_pix, y_pix, value (nJy per pixel), snr,
    truth_id (-1 if no component lies within match_radius pixels) and match_dist.

    base_image may be a memory map: one band is read and taken to the coadd PSF at a time.
    """
    truth = truth.copy()
    x0, y0 = origin
    ny, nx = base_image.shape[1:]
    truth["x_array"], truth["y_array"] = truth["x_pix"] - x0, truth["y_pix"] - y0
    truth["in_image"] = (truth["x_array"].between(-0.5, nx - 0.5, inclusive="left")
                         & truth["y_array"].between(-0.5, ny - 0.5, inclusive="left"))
    tree = cKDTree(truth[["x_array", "y_array"]].to_numpy(float))
    k = min(MAX_CANDIDATES, len(truth))
    peak_tables = []
    for i, band in enumerate(cfg["bands"]):
        clean = coadd_psf_band(np.asarray(base_image[i], np.float32), band, coadd, cfg)
        visits = int(coadd["n_visit"][band])
        snr = clean / np.sqrt(sky_sigma(band, visits, cfg) ** 2 + np.maximum(clean, 0) / visits)
        py, px = np.nonzero((clean == maximum_filter(clean, size=3, mode="nearest")) & (snr >= min_snr))
        dist, idx = tree.query(np.c_[px, py], k=k, distance_upper_bound=match_radius)
        dist, idx = dist.reshape(len(px), k), idx.reshape(len(px), k)
        best = np.argmin(np.where(np.isfinite(dist), dist + 1e-9 * idx, np.inf), axis=1)
        rows = np.arange(len(px))
        best_dist, best_idx = dist[rows, best], idx[rows, best]
        matched = np.isfinite(best_dist)
        peak_tables.append(pd.DataFrame(dict(
            band=band, x_array=px, y_array=py, x_pix=px + x0, y_pix=py + y0, value=clean[py, px], snr=snr[py, px],
            truth_id=np.where(matched, truth["truth_id"].to_numpy()[np.minimum(best_idx, len(truth) - 1)], -1),
            match_dist=np.where(matched, best_dist, np.nan))))

        owned = peak_tables[-1].query("truth_id >= 0").sort_values("snr", ascending=False)
        brightest = owned.drop_duplicates("truth_id").set_index("truth_id")
        truth[f"n_peaks_{band}"] = truth["truth_id"].map(owned.groupby("truth_id").size()).fillna(0).astype(int)
        truth[f"peak_{band}"] = truth[f"n_peaks_{band}"] > 0
        truth[f"peak_snr_{band}"] = truth["truth_id"].map(brightest["snr"])
        truth[f"peak_dist_{band}"] = truth["truth_id"].map(brightest["match_dist"])
        del clean, snr
    truth["n_peak_bands"] = truth[[f"peak_{band}" for band in cfg["bands"]]].sum(axis=1)
    return truth, pd.concat(peak_tables, ignore_index=True)
