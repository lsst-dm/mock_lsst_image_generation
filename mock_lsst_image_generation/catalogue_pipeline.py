"""Central mock-catalogue pipeline tying the model modules together.

build_mock_catalogue runs, for one seed:
  1. COSMOS donors -> empirical rest-frame SED/size PDF         (rest_frame_sed_sampling, donor_selection)
  2. clustered light-cone positions thinned by the evolving GSMF  (environment_sampling, cosmic_web_generation)
  3. type and stellar mass from the evolving GSMFs               (gsmf_sampling)
  4. sSFR from low-z COSMOS PDFs evolved to each redshift        (ssfr_sampling)
  5. cloned rest-frame SED -> observed ugrizy, bulge + disc      (rest_frame_sed_sampling, bulge_disc_decomposition)
  6. safety and render cuts
  7. interacting pairs, star-forming regions and tidal bridges    (tidal_stream_generation, sfregion_generation)
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from .bcg_sources import add_bcgs
from .bulge_disc_decomposition import add_bulge_disc_components, draw_bulge_to_total
from .config import CATALOGUE_STEM, CONFIG, N_CATALOGUES, PHYS, STAR_CONFIG, catalogue_splits
from .cosmology import build_cosmology_grids, trapz
from .donor_selection import rest_frame_sed_checks
from .environment_sampling import assign_clustered_positions, draw_web_positions
from .gsmf_sampling import build_gsmf_sampling_tables, draw_gsmf_masses
from .hubble_types import add_hubble_types
from .lsb_galaxies import add_lsb_galaxies
from .photometry import REST_COLS
from .rest_frame_sed_sampling import (RestFrameEmpiricalPDF, build_rest_frame_donor_table, project_to_observed_frame,
                                      sample_rest_frame_properties)
from .sfregion_generation import add_sfregions
from .stars import add_stars
from .ssfr_sampling import build_low_redshift_ssfr_pdf, draw_log_ssfr
from .tidal_stream_generation import add_tidal_streams, select_tidal_pairs
from .utils import combined_mask, print_cut_summary


class MockCatalogue(NamedTuple):
    """One realisation: the galaxy table, its star-forming regions, tidal-bridge blobs and interacting pairs, plus the
    fitted rest-frame PDF and the COSMOS donor table it was trained on (useful for diagnostics; None when read from
    disk)."""
    galaxies: pd.DataFrame
    sfregions: pd.DataFrame
    tidal_blobs: pd.DataFrame
    tidal_pairs: pd.DataFrame
    rest_frame_pdf: RestFrameEmpiricalPDF | None = None
    donors: pd.DataFrame | None = None


def draw_physical_catalogue(cfg, phys, rng, grids, ssfr_pdf):
    """Clustered light-cone positions, then type, stellar mass, sSFR, SFR, mass B/T and position angle."""
    tables = build_gsmf_sampling_tables(cfg, grids)
    n_expect = tables["n_expected"]
    n = int(rng.poisson(max(n_expect, 0.0))) if cfg["gsmf_poisson_counts"] else int(round(n_expect))
    print(f"Evolving GSMF expected N={n_expect:,.0f}; drawing N={n:,} down to logM={cfg['logm_min']:.1f}")
    pos = draw_web_positions(n, tables, cfg, grids, rng)
    if len(pos) == 0:
        return pd.DataFrame()

    z = pos["z"].to_numpy(float)
    p_sf = np.interp(z, tables["z"], tables["p_sf"], left=tables["p_sf"][0], right=tables["p_sf"][-1])
    kind = np.where(rng.random(len(pos)) < p_sf, "sf", "pa")
    sf = kind == "sf"
    logM = np.full(len(pos), np.nan)
    logM[sf] = draw_gsmf_masses(z[sf], "sf", tables, rng)
    logM[~sf] = draw_gsmf_masses(z[~sf], "pa", tables, rng)
    low_z_ssfr, ssfr_offset, logsSFR = draw_log_ssfr(logM, z, kind, ssfr_pdf, grids, cfg, phys, rng)
    logSFR = logM + logsSFR

    cat = pos.copy()
    cat["type"], cat["kind"], cat["p_star_forming_gsmf"] = np.where(sf, "star_forming", "passive"), kind, p_sf
    cat["logM"], cat["logsSFR_lowz_draw"], cat["logsSFR_delta_z"] = logM, low_z_ssfr, ssfr_offset
    cat["logsSFR"], cat["logSFR"] = logsSFR, logSFR
    cat["BT"] = draw_bulge_to_total(logM, logSFR, z, rng, phys)
    cat["pa_deg"] = rng.uniform(0.0, 180.0, len(cat))
    print("Physical catalogue type fractions:")
    print(cat["type"].value_counts(normalize=True).sort_index())
    print(cat[["z", "logM", "logsSFR", "logSFR"]].describe())
    return cat.reset_index(drop=True)


def compute_observables(cat, pdf, cfg, phys, rng, grids):
    """Clone rest-frame SEDs and sizes, project to observed ugrizy, split into bulge + disc, and number the galaxies."""
    rest = sample_rest_frame_properties(cat.assign(log1pz=np.log10(1.0 + cat["z"])), pdf, rng)
    if cfg["donor_mass_light_scaling"]:  # keep the donor's mass-to-light ratio: scale its luminosity to the mock mass
        rest[REST_COLS] = rest[REST_COLS].sub(2.5 * (rest["logM"] - rest["donor_logM"]), axis=0)
    obs = add_bulge_disc_components(project_to_observed_frame(rest, cfg, grids), cfg, phys, rng)
    obs.insert(0, "id", np.arange(len(obs)))
    return obs


def apply_mock_safety_cuts(mock, cfg):
    """Drop galaxies with broken rest-frame SEDs or non-finite / implausibly bright photometry, and renumber ids."""
    checks = rest_frame_sed_checks(mock, cfg)
    mag_cols = [f"mag_{b}_total" for b in cfg["out_bands"] if f"mag_{b}_total" in mock.columns]
    if mag_cols:
        checks.append(("finite observed mags", np.isfinite(mock[mag_cols]).all(axis=1)))
        mag_min = cfg["mock_observed_mag_min"]
        if mag_min is not None:
            checks.append((f"all observed mags >= {mag_min}", mock[mag_cols].ge(mag_min).all(axis=1)))
    flux_cols = [f"flux_{b}_total" for b in cfg["out_bands"] if f"flux_{b}_total" in mock.columns]
    if flux_cols:
        checks.append(("finite fluxes", np.isfinite(mock[flux_cols]).all(axis=1)))
    keep = combined_mask(checks, mock.index)
    if keep.all():
        return mock
    print_cut_summary(f"Mock safety cuts kept {int(keep.sum()):,} / {len(mock):,} galaxies before sfregions/tidal "
                      "streams", checks)
    out = mock.loc[keep].reset_index(drop=True)
    if "id" in out.columns:
        out["id"] = np.arange(len(out))
    return out


def apply_render_cuts(mock, cfg):
    """Keep galaxies worth rendering (mean r-band SB <= render_mu_r_max, r mag <= render_mag_r_max); renumber ids."""
    keep = np.ones(len(mock), bool)
    if cfg["render_mu_r_max"] is not None:
        keep &= np.isfinite(mock["sb_r_total"]) & (mock["sb_r_total"] <= cfg["render_mu_r_max"])
    if cfg["render_mag_r_max"] is not None:
        keep &= np.isfinite(mock["mag_r_total"]) & (mock["mag_r_total"] <= cfg["render_mag_r_max"])
    if keep.all():
        return mock
    print(f"Final render cuts kept {int(keep.sum()):,} / {len(mock):,}")
    mock = mock[keep].reset_index(drop=True)
    mock["id"] = np.arange(len(mock))
    return mock


def build_mock_catalogue(cosmos, cfg=None, phys=None, star_cfg=None):
    """Generate one mock-catalogue realisation from a COSMOS2025 table (see load_cosmos2025_catalogue).

    cfg / phys / star_cfg override the CONFIG / PHYS / STAR_CONFIG defaults (partial dicts are fine);
    cfg['seed'] fixes the realisation.
    """
    cfg, phys = {**CONFIG, **(cfg or {})}, {**PHYS, **(phys or {})}
    star_cfg = {**STAR_CONFIG, **(star_cfg or {})}
    rng, grids = np.random.default_rng(cfg["seed"]), build_cosmology_grids(cfg)

    donors = build_rest_frame_donor_table(cosmos, cfg, grids)
    print(f"Training rest-frame empirical PDF on {len(donors):,} galaxies")
    print(f"Median rest-edge distance: {donors['rest_edge_distance'].median():.3f} dex")
    print(donors[["z_cosmos", "logM", "logSFR", "logsSFR", "logRe_kpc", "ellipticity"]].describe())
    high_edge = donors["rest_edge_distance"] > cfg["warn_edge_distance"]
    if high_edge.mean() > 0.1:
        print(f"WARNING: {high_edge.mean():.1%} of training galaxies need large SED extrapolation.")
        print("         Add UV/NIR COSMOS bands to OBS_BAND_ALIASES if available.")
    pdf = RestFrameEmpiricalPDF(xcols=cfg["pdf_xcols"], xweights=cfg["pdf_xweights"], k=cfg["empirical_k"],
                                clip_percentiles=cfg["empirical_clip_percentiles"], seed=cfg["seed"]).fit(donors)
    zsel = grids["z"] >= cfg["z_min"]
    print(f'Survey area: {grids["area_deg2"]:.4f} deg^2; V = {trapz(grids["dVdz"][zsel], grids["z"][zsel]):,.0f} Mpc^3')

    cat = draw_physical_catalogue(cfg, phys, rng, grids, build_low_redshift_ssfr_pdf(cosmos, cfg))
    print(f"Drawn physical catalogue: {len(cat):,} galaxies")
    mock = apply_render_cuts(apply_mock_safety_cuts(compute_observables(cat, pdf, cfg, phys, rng, grids), cfg), cfg)
    if cfg["assign_positions_after_observables"]:
        mock = assign_clustered_positions(mock, cfg, grids, rng)

    # Before the star-forming regions, which follow the arms; own random stream, so nothing else changes.
    if cfg["hubble_types"]:
        mock = add_hubble_types(mock, np.random.default_rng([cfg["seed"], 2]))
    mock, tidal_pairs = select_tidal_pairs(mock, cfg, rng) if cfg["tidal_streams"] else (mock, pd.DataFrame())
    # After the tidal pairs, before the star-forming regions (which they get too); own random stream.
    if cfg["lsb_galaxies"]:
        mock = add_lsb_galaxies(mock, donors, cfg, phys, grids, np.random.default_rng([cfg["seed"], 4]))
    if cfg["bcgs"]:  # own random stream
        mock = add_bcgs(mock, cfg, np.random.default_rng([cfg["seed"], 5]))
    sfregions, tidal_blobs = pd.DataFrame(), pd.DataFrame()
    if cfg["sfregions"]:
        print("Adding star-forming regions...", flush=True)
        mock, sfregions = add_sfregions(mock, cfg, rng)
    if cfg["tidal_streams"]:
        print("Adding tidal streams...", flush=True)
        mock, tidal_blobs = add_tidal_streams(mock, tidal_pairs, cfg, rng)
    print("Projection edge distance percentiles:")
    print(np.nanpercentile(mock["projection_edge_distance"], [1, 16, 50, 84, 99]))
    if cfg["stars"]:  # own random stream, so the galaxies are identical with or without stars
        mock = add_stars(mock, cfg, np.random.default_rng([cfg["seed"], 1]), star_cfg)
    return MockCatalogue(mock, sfregions, tidal_blobs, tidal_pairs, pdf, donors)


def save_mock_catalogue(catalogue, out_csv):
    """Write <stem>.csv (galaxies) plus <stem>_sfregions.csv, <stem>_tidal.csv and <stem>_tidal_pairs.csv."""
    stem = str(Path(out_csv).with_suffix(""))
    for table, path, label in [(catalogue.galaxies, f"{stem}.csv", "galaxies"),
                               (catalogue.sfregions, f"{stem}_sfregions.csv", "sfregions"),
                               (catalogue.tidal_blobs, f"{stem}_tidal.csv", "tidal blobs"),
                               (catalogue.tidal_pairs, f"{stem}_tidal_pairs.csv", "tidal pairs")]:
        table.to_csv(path, index=False)
        print(f"Wrote {len(table):,} {label} -> {path}")


def load_mock_catalogue(csv_path):
    """Read a catalogue written by save_mock_catalogue (galaxies, star-forming regions, tidal blobs and tidal pairs)."""
    stem = str(Path(csv_path).with_suffix(""))

    def read(suffix):
        try:
            return pd.read_csv(f"{stem}{suffix}.csv")
        except (FileNotFoundError, pd.errors.EmptyDataError):  # companions are empty when a feature is switched off
            return pd.DataFrame()
    return MockCatalogue(read(""), read("_sfregions"), read("_tidal"), read("_tidal_pairs"))


def generate_mock_catalogues(cosmos, n_catalogues=N_CATALOGUES, out_dir=".", cfg=None, phys=None, stem=CATALOGUE_STEM,
                             numbered=False):
    """Build and save n independent realisations as <out_dir>/<stem>_<name>*.csv and return them by name.

    Names and seeds come from config.catalogue_splits: train, valid, calib, test, then extra1, extra2, ...; or 1, 2,
    3, ... if numbered.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalogues = {}
    for name, seed in catalogue_splits(n_catalogues, numbered).items():
        split_cfg = {**(cfg or {}), "seed": seed}
        npix = split_cfg.get("npix", CONFIG["npix"])
        out_csv = out_dir / f"{stem}_{name}.csv"
        print(f"\n===== Building '{name}' catalogue (seed={seed}, {npix}x{npix} px) -> {out_csv} =====", flush=True)
        catalogues[name] = build_mock_catalogue(cosmos, split_cfg, phys)
        save_mock_catalogue(catalogues[name], out_csv)
    return catalogues
