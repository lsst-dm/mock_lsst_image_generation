"""Default settings for mock-catalogue generation.

CONFIG covers the survey frame, COSMOS donor selection, sampling and feature switches; PHYS holds physical scaling
relations. build_mock_catalogue fills any key you leave out from these defaults, so overrides can be partial,
e.g. ``build_mock_catalogue(cosmos, dict(npix=2000, seed=7))``. A cut or limit set to None is disabled.
"""

from typing import Any

OUT_BANDS = ["u", "g", "r", "i", "z", "y"]

CONFIG: dict[str, Any] = dict(
    out_bands=OUT_BANDS,
    # COSMOS input columns (standardise_cosmos_columns also tries common aliases).
    z_col="zfinal", mass_col="mass_med", sfr_col="sfr_med", ssfr_col="ssfr_med", sfr_is_log=True,
    re_arcsec_col="radius_sersic_arcsec", ellipticity_col="ellipticity",
    # Square survey frame, redshift/mass range, cosmology and RNG seed.
    npix=5000, pixscale=0.2, z_min=0.0, z_max=6.0, logm_min=7.0, logm_max=12.0, H0=70.0, Om0=0.3, seed=42,
    # Continuous COSMOS-style GSMF sampling. Redshifts come from clustered light-cone sites, which are thinned by the
    # total evolving number density.
    gsmf_n_z=700, gsmf_n_m=900, gsmf_z_eval_min=0.2, gsmf_z_eval_max=5.5, gsmf_poisson_counts=True,
    gsmf_local_anchor_z=(0.35, 0.6),  # GAMA-anchored below the first z, fading out by the second (None = off)
    gsmf_position_oversample=4,
    # Low-z COSMOS2025 empirical sSFR PDFs, split into SF/passive samples and evolved to each galaxy's redshift.
    ssfr_pdf_z_min=0.0, ssfr_pdf_z_max=0.3, ssfr_split=-10.0, ssfr_pdf_k=160, ssfr_reference_z=0.15,
    ssfr_min=-15.5, ssfr_max=-6.5,
    # Empirical rest-frame PDF p(SED, Re, ellipticity | logM, logsSFR) built from COSMOS donors.
    pdf_xcols=["logM", "logsSFR"], pdf_xweights=None, empirical_k=256, empirical_clip_percentiles=(0.5, 99.5),
    # Scale each cloned SED by 10^(logM - donor logM), keeping the donor's mass-to-light ratio and colours.
    donor_mass_light_scaling=False,
    min_training_bands=4, allow_sed_extrapolation=True, warn_edge_distance=0.35, max_rest_edge_distance=0.55,
    # COSMOS donor quality cuts, restricted to bright, reliable HSC-r model magnitudes.
    cosmos_galaxies_only=True,  # LePhare galaxies only: no QSO (or star) donors
    # Optional size-quality cut, off: keep only donors whose COSMOS Sersic Re lies inside the Kron aperture
    # (Re <= value x kron1_a). Off by default because it also removes most of the extended donors.
    cosmos_max_sersic_re_over_kron=None,
    cosmos_training_z_max=6.0, cosmos_training_logm_min=7.0, cosmos_training_logm_max=12.0,
    cosmos_training_logssfr_min=-15.5, cosmos_training_logssfr_max=-7.0,
    cosmos_training_r_mag_min=20.0, cosmos_training_r_mag_max=24.0, cosmos_training_r_magerr_max=0.2,
    cosmos_training_ellipticity_min=0.01, cosmos_training_ellipticity_max=0.95,
    cosmos_training_mag_max=32.0, cosmos_training_require_all_obs_bands=True,
    cosmos_apparent_colour_limits=[("obs_u", "obs_g", -3.0, 6.0), ("obs_g", "obs_r", -3.0, 6.0),
                                   ("obs_r", "obs_i", -3.0, 6.0), ("obs_i", "obs_z", -3.0, 6.0),
                                   ("obs_z", "obs_y", -3.0, 6.0)],
    # Reject sharp observed-frame SED zig-zags, e.g. a single depressed r band making sources look purple in gri RGB.
    cosmos_observed_colour_curvature_max=1.0,
    # HSC g/r/i colour quality: colour errors and a robust g-r/r-i colour-plane ellipse.
    cosmos_gr_ri_colour_error_max=0.15, cosmos_gr_ri_colour_plane_sigma=4.0, cosmos_gr_ri_colour_plane_min_fit=300,
    # Very large galaxies look unphysical when the donor gives them extreme flattening: cap only the large-Re tail.
    large_re_ellipticity_cap_arcsec=1.0, large_re_ellipticity_cap=0.8,
    # Rest-frame SED sanity cuts, deliberately broad: reject broken/interpolated SEDs without forcing a narrow locus.
    rest_abs_mag_min=-30.0, rest_abs_mag_max=5.0,
    rest_abs_mag_faint_limits=dict(M0900=12.0, M1216=8.0, M1500=8.0, M1900=6.0, M2200=6.0, M2500=6.0,
                                   Mu=5.0, Mg=5.0, Mr=5.0, Mi=5.0, Mz=5.0, My=5.0),
    rest_colour_curvature_max=1.0,  # zig-zags between adjacent rest-frame nodes
    rest_colour_limits=[("M0900", "M1216", -0.5, 8.0), ("M1216", "M1500", -2.5, 5.0), ("M1500", "M1900", -3.0, 5.0),
                        ("M1900", "M2200", -3.0, 4.0), ("M2200", "M2500", -3.0, 4.0), ("M2500", "Mu", -3.0, 5.0),
                        ("Mu", "Mg", -2.5, 4.0), ("Mg", "Mr", -2.5, 4.0), ("Mr", "Mi", -2.0, 3.0),
                        ("Mi", "Mz", -2.0, 3.0), ("Mz", "My", -2.0, 3.0), ("M1500", "Mr", -3.0, 6.0),
                        ("Mu", "Mr", -2.0, 4.0), ("Mg", "Mi", -1.5, 3.5)],
    mock_observed_mag_min=10.0,
    # Surface-brightness definition and final cuts on what is worth rendering.
    sb_re_mode="circularized", render_mu_r_max=30.0, render_mag_r_max=None,
    # Bulge/disc structure.
    bulge_re_frac=0.20,
    bulge_disc_q_coupling=0.75,  # stronger = bulge follows an edge-on disc more
    bulge_disc_q_offset=0.16,  # bulge stays rounder than the disc by this amount in q
    bulge_q_min=0.18, bulge_q_max=0.95,
    bulge_edgeon_q0=0.60,  # coupling only becomes important below this disc q
    # Positions: one coherent light-cone cosmic web, calibrated on a 100 Mpc toy cube. With
    # assign_positions_after_observables the finished galaxies are re-placed onto web sites by environment score.
    clustered_positions=True, assign_positions_after_observables=True,
    # Joint mass / star formation / environment: each galaxy's environment is drawn from its class (red / blue x low /
    # high surface brightness as T24 selected them; environment_sampling.reference_classes) as (P(structure), P(cluster
    # |
    # structure)). For a class with structure fraction f, w(theta) ~ f^2 w_structure (field sites are unclustered), so
    # f = sqrt(A_T24 / A_structure) with A_structure = 0.67 at 1 deg (mock, z < 0.1) and A_T24 the Thuruthipilly et al.
    # 2024 (Table 4) amplitudes: red HSB 0.245, blue HSB 0.065, red LSB 0.671 (in cluster outskirts), blue LSB 0.051.
    environment_by_class=dict(red_hsb=(0.69, 0.3), blue_hsb=(0.40, 0.0), red_lsb=(1.0, 0.9), blue_lsb=(0.41, 0.05)),
    web_parent_pad_mpc=35.0, web_candidate_oversample=8,
    # Draw cluster/filament sites only from web structure within web_cone_reach_mpc of the cone until the design
    # fractions (web_f_cluster, web_f_filament) hold inside the cone; False = original draw-everywhere-then-clip,
    # which left the narrow cone ~97 % field (see cosmic_web_generation.fill_cone_structure_sites).
    web_sites_fill_cone=True, web_cone_reach_mpc=25.0,
    web_n_nodes_100mpc=60, web_node_boost=1.0, web_f_cluster=0.15, web_f_filament=0.25, web_r_cluster=5.0,
    web_sig_filament=1.2, web_core_frac=0.30, web_k_extra=3, web_k_connect=16, web_max_extra=22.0,
    web_n_super=8, web_super_sigma=18.0, web_frac_lss=0.30,  # large-scale modulation of the node field
    # web_f_cluster, web_f_filament and web_frac_lss are tuned so that the ensemble-mean projected correlation
    # function of log M >= 9.5 galaxies at 0.2 < z < 1.2 matches Zehavi et al. (2005): r0 = 5.59 / h Mpc,
    # gamma = 1.84 (16 realisations: gamma = 1.85, within 0.7-1.3 of the power law at 0.1-6.5 Mpc).
    web_assign_bin_mpc=15.0, web_site_keep_bias=0.08, env_assign_scatter=0.75,  # site assignment in narrow shells
    # Star-forming regions.
    sfregions=True, sfregion_mean_single_u_frac=0.0035, sfregion_max_r_re=2.0, sfregion_central_hole_re=1.0,
    sfregion_flux_radius_bias=0.0, sfregion_flux_scatter=0.35, sfregion_sigma_floor_arcsec=0.01,
    sfregion_n_max_per_gal=64,
    # Tidal bridges between interacting pairs. Targets are fractions of galaxies per environment, not per-pair
    # probabilities; the candidate pool is loose enough for this light-cone mock.
    tidal_streams=True, tidal_target_field=0.02, tidal_target_filament=0.08, tidal_target_cluster=0.04,
    tidal_pair_rp_min_mpc=0.02, tidal_pair_rp_max_mpc=0.60, tidal_pair_dchi_max_mpc=25.0,
    tidal_pa_scatter_deg=18.0, tidal_anchor_radius_re=2.0, tidal_flux_frac_min=0.03, tidal_flux_frac_max=0.08,
    tidal_column_sig_re_frac_min=0.28, tidal_column_sig_re_frac_max=0.44,
    tidal_column_sig_pix_min=4.4, tidal_column_sig_pix_max=9.6,
    tidal_n_curve=9, tidal_n_straight=11, tidal_curve_strength=0.7, tidal_inward_pull=0.45,
    # Star-forming regions and tidal features only go on resolved galaxies: resolved_re_col > PSF FWHM in
    # resolved_psf_band.
    psf_fwhm_arcsec=dict(u=1.16, g=1.11, r=1.05, i=1.01, z=0.97, y=0.95), resolved_psf_band="r",
    resolved_re_col="re_total_arcsec",
    # Bright stars (see stars.py and STAR_CONFIG): added to the catalogue as rows with type "star".
    stars=True,
    # Hubble types drawn from published mass functions, with the arm / bar / clump parameters they imply
    # (hubble_types.py).
    hubble_types=True,
    # Extended dwarf irregulars, UDGs and almost-dark galaxies (lsb_galaxies.py), added on top of the realistic
    # population in deliberately generous numbers (per deg^2) so a detector sees enough of them; each population is
    # drawn within its observed ranges of redshift, Re (kpc) and central g-band surface brightness mu0_g.
    # BCG-like core-Sersic sources (BCG core + extended envelope; bcg_sources.py, ICL_DoubleSersic_Injection.ipynb):
    # Re (arcsec) log-uniform and mean r-band SB within the half-light area mu_r uniform in these ranges.
    # The rates keep these classes rare enough not to crowd a frame (their footprints cover ~2% of it) while a
    # training frame of ~0.6 deg^2 (14000 x 14000 px; bin.src/generate_large_training_frame.py) holds several hundred
    # of each: ~480 UDGs, ~480 extended dIrrs, ~240 almost-dark galaxies, ~24 BCGs. A 5000 x 5000 px frame holds ~1/8
    # of that. Almost-dark galaxies are drawn 4x as often as the others' rate would suggest, being the rarest.
    bcgs=True, bcg_per_deg2=40, bcg_re_arcsec=(5.0, 30.0), bcg_mu_r=(20.0, 25.0),
    lsb_galaxies=True,
    lsb_populations=dict(
        extended_dirr=dict(per_deg2=800, z=(0.005, 0.06), re_kpc=(1.0, 3.0), mu0_g=(22.5, 24.0), quiescent_frac=0.0,
                           donors="star_forming", irregularity=(0.5, 0.1)),
        udg=dict(per_deg2=800, z=(0.02, 0.06), re_kpc=(1.5, 5.0), mu0_g=(24.0, 27.0), quiescent_frac=0.25,
                 donors="star_forming", irregularity=(0.5, 0.1)),
        almost_dark=dict(per_deg2=400, z=(0.003, 0.03), re_kpc=(1.5, 3.5), mu0_g=(26.0, 28.0), quiescent_frac=0.0,
                         donors="bluest", irregularity=(0.8, 0.05)),
    ),
)

# Stars, calibrated on the stars of a real LSSTCam deep coadd (StarSimulation.ipynb). Only g, r and i were calibrated:
# u uses g's light-profile template, z and y use i's.
STAR_CONFIG: dict[str, Any] = dict(
    density_per_deg2=5600, mag_range=(9.0, 23.0), count_slope=0.185,  # field stars to r = 23; dlog N / dm
    bright_per_deg2=100, bright_mag_range=(10.0, 15.0),  # extra bright stars, so training sees spikes and saturation
    colour_match_mag=1.0,  # colours are resampled from field stars within ~this many mag
    red_g_r=1.1, red_drop_per_mag=0.3,  # red (M-dwarf) colours fade out brighter than the field's brightest stars
    # Light profile as fractions of total flux: seeing core, Moffat halo (beta, FWHM") with a soft outer edge and
    # radial streaks, and diffraction spikes (on-axis brightness per pixel at 10", summed over an image's lines).
    halo_basis=[(2.5, 5.0), (2.0, 12.0)],
    halo_fraction=dict(u=[0.0293, 0.0072], g=[0.0293, 0.0072], r=[0.0414, 0.0049], i=[0.0426, 0.0046],
                       z=[0.0426, 0.0046], y=[0.0426, 0.0046]),
    spike_at_10arcsec=dict(u=6.5e-7, g=6.5e-7, r=1.33e-6, i=7.4e-7, z=7.4e-7, y=7.4e-7),
    spike_rotations=(1, 3),  # camera rotations per band in one image; each gives two perpendicular spike lines
    arm_scatter=0.38, spike_taper_arcsec=2.0,
    # Per-star ranges (uniform unless noted).
    spike_scale_sigma=0.3,  # lognormal
    spike_core_arcsec=(0.5, 1.5), spike_slope=(2.0, 2.4), spike_width=(1.1, 1.7),  # width in units of the PSF FWHM
    halo_cut_arcsec=(20.0, 35.0), streak_strength=(0.1, 0.35),
    saturation_r=(6300.0, 19100.0),  # nJy/pixel, log-uniform; other bands scale by saturation_ratio
    saturation_ratio=dict(u=1.92, g=0.98, r=1.0, i=0.84, z=1.16, y=2.59),
    # Saturated cores: flat rows whose values scatter around the saturation level (coloured stripes in gri).
    row_scatter=0.15, row_correlation=2.0, core_stretch=(1.0, 1.25), core_shift_px=1.5,
)

PHYS: dict[str, Any] = dict(
    passive_logssfr_z_slope=0.4,  # passive sSFR evolution about the low-z reference (SF follows the main sequence)
    bt_conc=6.0,  # Beta-distribution concentration of the B/T scatter
    bulge_ba=dict(mean=0.80, sig=0.08, lo=0.5, hi=0.95),  # intrinsic spheroid axis ratio
    bulge_ml_ratio=dict(u=3.0, g=2.5, r=1.8, i=1.5, z=1.35, y=1.25),  # (M/L)_bulge / (M/L)_disc per band
)

# Catalogues are statistically identical, independent realisations of the same frame: only the seed differs.
CATALOGUE_NAMES = ["train", "valid", "calib", "test"]  # then extra1, extra2, ...
N_CATALOGUES = len(CATALOGUE_NAMES)
CATALOGUE_STEM = "mock_catalogue"

IMAGE_CONFIG: dict[str, Any] = dict(
    bands=OUT_BANDS, pixscale=CONFIG["pixscale"], zeropoint=31.4,  # catalogue fluxes are nJy, AB zeropoint 31.4
    # Nominal 10-year median-seeing PSF FWHM per band (arcsec, PSTN-054). Also sets the band-to-band seeing ratios.
    nominal_fwhm=CONFIG["psf_fwhm_arcsec"],
    # 10-year 5-sigma point-source depth (AB) and number of visits per band (Ivezic et al., LSST survey strategy).
    depth_10yr=dict(u=26.2, g=27.4, r=27.6, i=26.9, z=26.1, y=24.8),
    visits_10yr=dict(u=56, g=80, r=184, i=185, z=160, y=160),
    # The scene is rendered once at this narrow PSF (arcsec) and broadened to each target, so it must be narrower
    # than every target PSF.
    base_fwhm=0.45,
    # Depth axis: survey epoch -> fraction of the 10-year visits.
    epochs={"1m": 1.0 / 120.0, "6m": 0.05, "1y": 0.10, "3y": 0.30, "5y": 0.50, "8y": 0.80, "10y": 1.00},
    # Seeing axis: r-band coadd PSF FWHM (arcsec); other bands follow the nominal ratios with a small random jitter.
    fwhm_grid_r=[0.7, 0.9, 1.1, 1.3, 1.5, 2.0], fwhm_ratio_jitter=0.03,
    # Catalogues whose coadds are not saved by default: training code rebuilds them on the fly from the base image.
    on_the_fly_catalogues=["train", "valid"],
    min_stamp_pix=25,  # smallest GalSim stamp; there is deliberately no maximum, so no flux is clipped
    # Image canvas. None: the canvas spans the galaxy centres, so its size and origin depend on the realisation. An
    # integer N: the canvas is exactly frame pixels [0, N) on both axes with origin (0, 0), e.g. the catalogue's npix,
    # so images line up with a fixed patch grid. Light from sources off the canvas is clipped at its edges either way.
    canvas_npix=None,
    # Spiral arms, bars and irregular clumps (galaxy_structure.py) are drawn on discs at least this large (arcsec).
    structure_min_re_arcsec=0.4,
    # PSF model. "gaussian": every source sees a Gaussian PSF and only stars get a halo and diffraction spikes; the
    # training grid needs it, since one base render is broadened to every seeing. "moffat" (single coadds only): a
    # Moffat core of FWHM moffat_fwhm and index moffat_beta per band, each jittered by fwhm_ratio_jitter /
    # moffat_beta_jitter, with the stars' halo and spikes (STAR_CONFIG) as part of the PSF, so galaxies and stars see
    # the same PSF. psf_wings fixes the halo and spike shape that STAR_CONFIG otherwise draws per star.
    psf_model="gaussian",
    moffat_fwhm=dict(u=1.25, g=1.05, r=0.85, i=0.80, z=0.95, y=1.10),
    moffat_beta=dict(u=2.8, g=3.2, r=3.6, i=3.6, z=3.3, y=3.0), moffat_beta_jitter=0.05,
    psf_wings=dict(half_pix=300, halo_cut_arcsec=27.5, streak_strength=0.2, spike_core_arcsec=1.0, spike_slope=2.2,
                   spike_width=1.4),
    # Side of the PSF image saved with each coadd (pixels, odd).
    psf_stamp_pix=41,
)


def catalogue_index(name):
    """Position of a catalogue in the train, valid, calib, test, extra1, extra2, ... sequence; numbered names 1, 2, 3, ...
    count along the same sequence, so catalogue 1 is the same realisation as train."""
    if name.isdigit() and int(name) >= 1:
        return int(name) - 1
    if name in CATALOGUE_NAMES:
        return CATALOGUE_NAMES.index(name)
    if name.startswith("extra") and name[5:].isdigit() and int(name[5:]) >= 1:
        return len(CATALOGUE_NAMES) + int(name[5:]) - 1
    raise ValueError(f"Unknown catalogue name {name!r}: expected {', '.join(CATALOGUE_NAMES)}, extra1, extra2, ... "
                     "or 1, 2, 3, ...")


def catalogue_splits(n_catalogues=N_CATALOGUES, numbered=False):
    """{name: seed} for n realisations: train, valid, calib, test (seeds 42, 101, 202, 303), then extra1, extra2, ...
    (seeds 404, 505, ...); or, if numbered, 1, 2, 3, ... with the same seeds. Realisation k always has the same name
    and seed, so adding more never changes the rest."""
    if n_catalogues < 1:
        raise ValueError(f"n_catalogues must be at least 1, got {n_catalogues}")
    if numbered:
        names = [str(k) for k in range(1, n_catalogues + 1)]
    else:
        names = CATALOGUE_NAMES[:n_catalogues]
        names += [f"extra{k}" for k in range(1, n_catalogues - len(names) + 1)]
    return {name: 42 if k == 0 else 101 * k for k, name in enumerate(names)}
