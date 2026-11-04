"""Render each mock catalogue once and write the coadds needed for training and evaluation.

For catalogue <name>, <out_dir>/<name>/ holds:
  base_clean_signal.npy           noise-free render at the narrow base PSF (float32, band x y x, nJy per pixel)
  base_<component>_signal.npy     the light of the star-forming regions, the tidal features and the stars' diffraction
                                  spikes alone (component = sfregions, tidal, spikes), on the same grid: truth for
                                  learning where each phenomenon is
  base_meta.json                  bands, canvas origin in frame pixels, shape, pixel scale, zeropoint and base PSF
  coadd_manifest.json             every coadd: visits and PSF FWHM per band, noise seed, and whether it was saved
  <name>_<coadd>_signal.npy       saved coadds (nJy per pixel) ...
  <name>_<coadd>_variance.npy     ... their per-pixel variance ...
  <name>_<coadd>_mask.npy         ... their mask (uint8, bits named in base_meta.json 'mask_planes': SAT marks
                                  the saturated star cores) ...
  <name>_<coadd>_psf.npy          ... and their PSF (float32, band x n x n, n = psf_stamp_pix; psf.coadd_psf_image)

With cfg['psf_model'] 'moffat' (single mode only), the base render is drawn with the coadd's Moffat core and the coadd
adds the PSF's halo and diffraction spikes to every source (psf.py). base_spikes_signal then holds the light of the
PSF's spikes in the coadd, from every source.

Modes:
  training   (default) every (epoch, seeing) coadd goes in the manifest; all are saved except for train/valid, whose
             coadds training code rebuilds on the fly from the base render and manifest, so they cost no disk.
  all        every coadd is saved for every catalogue.
  single     one coadd, saved for every catalogue: pass epoch (e.g. '10y', at the nominal PSF), or n_exp and psf_fwhm
             (r-band visits and PSF FWHM); see coadd_synthesis.single_coadd.
"""

import gc
import json
from pathlib import Path

import numpy as np

from .catalogue_pipeline import load_mock_catalogue
from .coadd_synthesis import MASK_PLANES, coadd_grid, single_coadd, synthesise_coadd
from .config import CATALOGUE_STEM, IMAGE_CONFIG, catalogue_index
from .galaxy_rendering import render_catalogue, split_stars
from .psf import add_wings, coadd_psf_image, moffat_core

MODES = ("training", "all", "single")


def available_catalogues(catalogue_dir, stem=CATALOGUE_STEM):
    """Names of the catalogues saved in catalogue_dir, in train, valid, calib, test, extra1, ... (or 1, 2, ...) order."""
    names = []
    for path in Path(catalogue_dir).glob(f"{stem}_*.csv"):
        name = path.stem[len(stem) + 1:]
        try:
            names.append((catalogue_index(name), name))
        except ValueError:  # the _sfregions / _tidal / _tidal_pairs companions
            pass
    return [name for _, name in sorted(names)]


def write_json(path, content):
    with Path(path).open("w") as file:
        json.dump(content, file, indent=2)


def generate_catalogue_images(catalogue, name, out_dir, mode="training", cfg=None, n_workers=1, epoch=None, n_exp=None,
                              psf_fwhm=None):
    """Render one catalogue and write its base image, manifest and coadds (see the module docstring)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    cfg = {**IMAGE_CONFIG, **(cfg or {})}
    moffat = cfg["psf_model"] == "moffat"
    if cfg["psf_model"] not in ("gaussian", "moffat") or (moffat and mode != "single"):
        raise ValueError(f"psf_model must be 'gaussian', or 'moffat' in single mode; got {cfg['psf_model']!r} in "
                         f"{mode} mode")
    index = catalogue_index(name)
    # Build the coadd list first, so a bad epoch / n_exp / psf_fwhm fails before the slow render.
    coadds = single_coadd(index, cfg, epoch, n_exp, psf_fwhm) if mode == "single" else coadd_grid(index, cfg)
    for band in [] if moffat else cfg["bands"]:
        if min(cfg["fwhm_grid_r"]) * cfg["nominal_fwhm"][band] / cfg["nominal_fwhm"]["r"] <= cfg["base_fwhm"]:
            raise ValueError(f"base_fwhm {cfg['base_fwhm']} must be narrower than every target PSF ({band} band)")
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if moffat:
        (coadd,) = coadds.values()
        base_psf = {band: moffat_core(coadd, band) for band in cfg["bands"]}
        print(f"\n===== [{name}] rendering at the Moffat core PSF, FWHM = {coadd['fwhm_arcsec']}\" =====", flush=True)
    else:
        base_psf = {band: cfg["base_fwhm"] for band in cfg["bands"]}
        print(f"\n===== [{name}] rendering at the base PSF, FWHM = {cfg['base_fwhm']}\" =====", flush=True)
    components = {}
    base_image, origin = render_catalogue(catalogue, base_psf, cfg, n_workers, star_seed=index,
                                          components_out=components)
    if moffat:
        components["spikes"] = np.stack([add_wings(base_image[i], band, coadd, cfg, spikes_only=True)
                                         for i, band in enumerate(cfg["bands"])])
    stars = split_stars(catalogue.galaxies)[1]
    np.save(out_dir / "base_clean_signal.npy", base_image)
    for component, light in components.items():
        np.save(out_dir / f"base_{component}_signal.npy", light)
    del components
    write_json(out_dir / "base_meta.json", dict(split=name, origin=list(origin), bands=cfg["bands"],
                                                base_fwhm=None if moffat else cfg["base_fwhm"],
                                                shape=list(base_image.shape), ps=cfg["pixscale"], zp=cfg["zeropoint"],
                                                psf_model=cfg["psf_model"],
                                                mask_planes={plane: 1 << bit for plane, bit in MASK_PLANES.items()}))

    save = mode != "training" or name not in cfg["on_the_fly_catalogues"]
    manifest_coadds = {}
    for key, coadd in coadds.items():
        if save:
            signal, variance, mask = synthesise_coadd(base_image, coadd, cfg, stars, origin, return_mask=True)
            np.save(out_dir / f"{name}_{key}_signal.npy", signal)
            np.save(out_dir / f"{name}_{key}_variance.npy", variance)
            np.save(out_dir / f"{name}_{key}_mask.npy", mask)
            np.save(out_dir / f"{name}_{key}_psf.npy", coadd_psf_image(coadd, cfg))
            del signal, variance, mask
            gc.collect()
        manifest_coadds[key] = {**{k: v for k, v in coadd.items() if k != "noise_seed"},
                                   "fwhm_arcsec": {band: round(v, 4) for band, v in coadd["fwhm_arcsec"].items()}}
    write_json(out_dir / "coadd_manifest.json", dict(split=name, base_fwhm=cfg["base_fwhm"], bands=cfg["bands"],
                                                     materialised=save, combos=manifest_coadds))
    print(f"[{name}] {len(coadds)} coadds in the manifest, {len(coadds) if save else 0} saved -> {out_dir}")


def render_single_coadd(catalogue, index, cfg=None, n_workers=1, epoch=None, n_exp=None, psf_fwhm=None):
    """Render one catalogue into a single coadd in memory, without writing any files.

    This is the body of ``generate_catalogue_images`` in single mode with nothing saved to disk: it builds the base
    render for one (epoch, PSF) combination and synthesises the coadd, returning the pixel products and the base image
    needed for truth tables. ``index`` seeds the stars and the coadd noise, as ``catalogue_index(name)`` does for the
    file-based path. Pass ``epoch`` (e.g. ``'10y'``), or ``n_exp`` with ``psf_fwhm``; see
    `~coadd_synthesis.single_coadd`.

    Returns ``(key, settings, bands, origin, base_image, signal, variance, mask, psf)``: the coadd key and its manifest
    settings, the band order, the frame-pixel origin of ``[:, 0, 0]``, the noise-free base render (float32, band y x)
    and the coadd's signal, variance, mask and PSF cutout.
    """
    cfg = {**IMAGE_CONFIG, **(cfg or {})}
    moffat = cfg["psf_model"] == "moffat"
    if cfg["psf_model"] not in ("gaussian", "moffat"):
        raise ValueError(f"psf_model must be 'gaussian' or 'moffat', got {cfg['psf_model']!r}")
    (key, coadd), = single_coadd(index, cfg, epoch, n_exp, psf_fwhm).items()
    if moffat:
        base_psf = {band: moffat_core(coadd, band) for band in cfg["bands"]}
    else:
        base_psf = {band: cfg["base_fwhm"] for band in cfg["bands"]}
    base_image, origin = render_catalogue(catalogue, base_psf, cfg, n_workers, star_seed=index)
    stars = split_stars(catalogue.galaxies)[1]
    signal, variance, mask = synthesise_coadd(base_image, coadd, cfg, stars, origin, return_mask=True)
    psf = coadd_psf_image(coadd, cfg)
    return key, coadd, cfg["bands"], origin, base_image, signal, variance, mask, psf


def generate_mock_images(catalogue_dir, out_dir, names=None, mode="training", cfg=None, n_workers=1, epoch=None,
                         n_exp=None, psf_fwhm=None, stem=CATALOGUE_STEM):
    """Generate images for the named catalogues in catalogue_dir (default: every catalogue there)."""
    available = available_catalogues(catalogue_dir, stem)
    if not available:
        raise FileNotFoundError(f"No {stem}_<name>.csv catalogues in {catalogue_dir}: run generate_mock_catalogues.py")
    missing = sorted(set(names or []) - set(available))
    if missing:
        raise FileNotFoundError(f"Catalogues {missing} not found in {catalogue_dir}; available: {available}")
    for name in names or available:
        catalogue = load_mock_catalogue(Path(catalogue_dir) / f"{stem}_{name}.csv")
        generate_catalogue_images(catalogue, name, out_dir, mode, cfg, n_workers, epoch, n_exp, psf_fwhm)
