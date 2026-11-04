# mock_lsst_image_generation

This is a fork of the code created by Tom Sedgwick at
[tmsedgwick/mock_lsst_image_generation](https://github.com/tmsedgwick/mock_lsst_image_generation).
The design of the code and its implementation is all his work; in DM we continue to build on it. The package
itself depends only on numpy, scipy, pandas and GalSim. The adapters that turn these mocks into Rubin Science
Pipelines data products (afw exposures, a butler repository, a generator PipelineTask) live in a separate package,
`mock_gen` (importable as `lsst.mock.gen`), which imports this one.

Mock LSST-like galaxy catalogues and images for training and testing source detection and deblending, built from
physical models and COSMOS2025 data. Every galaxy is a known truth entry, so any detector's completeness and purity can
be measured exactly.

Two commands: one builds the catalogues, the other renders them into images. Both require their output folder; keep
it outside the repo, since a full image set can be ~200 GB.

```bash
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images
```

## What the catalogue models

- **Positions:** galaxies placed on a cosmic web (clusters, filaments, field) along a light cone
- **Type and stellar mass:** evolving star-forming and quiescent stellar mass functions
- **Star formation, colours and sizes:** drawn from similar real galaxies in COSMOS2025 and shifted to each mock
  galaxy's redshift, giving LSST ugrizy photometry
- **Structure:** bulge + disc
- **Extras:** star-forming regions and tidal bridges between interacting pairs
- **Stars:** bright LSST-like stars with band-dependent diffraction spikes, halos and saturated cores, calibrated on
  a real LSSTCam deep coadd (`stars.py`, settings in `STAR_CONFIG`)
- **Hubble types:** each galaxy gets a type (E0-E7, S0, Sa-Sc, barred SB0-SBc, Irr), drawn with probabilities
  from published morphological mass functions (Huertas-Company et al. 2016), T-type ratios (Nair & Abraham 2010)
  and bar fractions (Erwin 2018; Melvin et al. 2014). Resolved discs are drawn with the spiral arms, bar or
  irregular clumps of their type, as light moved within the disc so band fluxes are unchanged (`hubble_types.py`,
  `galaxy_structure.py`)
- **BCG-like sources:** core-Sersic envelopes (BCG core + extended diffuse light; Graham et al. 2003 profile
  with the parameters and photometry of `ICL_DoubleSersic_Injection.ipynb`), 40 per deg^2 (`bcg_sources.py`; `bcgs`,
  `bcg_per_deg2`, `bcg_re_arcsec`, `bcg_mu_r` in `CONFIG`); rows have `profile` = `core_sersic`
- **Extended dwarfs, UDGs and almost-dark galaxies:** added on top of the realistic population in generous but
  uncrowded numbers (their footprints cover ~2% of a frame; `lsb_galaxies.py`, rates in `CONFIG["lsb_populations"]`:
  800 UDGs and 800 extended dIrrs per deg^2, almost-dark galaxies 400). A 0.6 deg^2 training frame (below, *Large
  training frames*) then holds several hundred of each:
  extended star-forming dIrrs, ultra-diffuse galaxies (van Dokkum et al. 2015 definition; mostly blue and irregular
  in the field, Leisman et al. 2017) and almost-dark galaxies with barely visible, very blue, irregular light
  (ALFALFA examples AGC 229385 and AGC 229101), each drawn within its observed size and surface-brightness range

`catalogue_pipeline.build_mock_catalogue` ties these together. All settings live in `config.py` (`CONFIG`, `PHYS`,
and `IMAGE_CONFIG` for the images); pass a partial dict to override any of them.

## Repository layout

| Path | Contents |
|---|---|
| `mock_lsst_image_generation/` | The package: one module per model ingredient, `config.py` for all settings |
| `scripts/generate_mock_catalogues.py` | Command: build the catalogues |
| `scripts/generate_mock_images.py` | Command: render the catalogues into images |
| `scripts/generate_large_training_frame.py` | Command: one large training frame (e.g. 14000 x 14000 px), built as independent parts and stitched |
| `data/cosmos2025_subset.npz` | Shipped COSMOS2025 input (see below) |
| `tests/` | pytest suite, run by GitHub Actions on every push |

## Install

Create a virtual environment inside the repo and install the pinned dependencies plus the package itself:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions the pipeline was verified with (including GalSim), so catalogues and
images are reproducible. To use your own versions instead, run `pip install -e .`. Point your editor's Python interpreter at
`.venv/bin/python` so it can resolve the imports.

The repo ships `data/cosmos2025_subset.npz` (17 MB), used by default. It holds only the columns the pipeline reads and
the 87k rows (of 784k) that can pass its COSMOS cuts, and gives byte-identical catalogues to the full
`cosmos2025_cat.npz` (~1.5 GB) at the default settings. If you loosen the COSMOS training/sSFR-PDF cuts in `CONFIG`,
pass the full catalogue instead.

## Generate catalogues

```bash
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues   # add --cosmos /path/to/cosmos2025_cat.npz for the full table
```

This builds four catalogues by default: `train`, `valid`, `calib` and `test`. Pass `--n-catalogues N` to build any
number; catalogues beyond the fourth are named `extra1`, `extra2`, and so on. Each catalogue is an independent
realisation of a 5000 × 5000 pixel frame (0.2″ pixels), and only the seed differs. Catalogue k always gets the same
name and seed, so asking for more never changes the ones you already have.

If you are not training a model, `--numbered` names them `1`, `2`, `3`, ... instead (same seeds, so `1` is the same
realisation as `train`).

Each catalogue writes four CSVs to `--out-dir`:

| File | Contents |
|---|---|
| `mock_catalogue_<name>.csv` | Galaxies: position, redshift, environment, physical properties, and bulge/disc/total photometry and structure. Stars are extra rows with `type` = `star` |
| `…_sfregions.csv` | Star-forming regions: parent id, offsets, width, per-band flux |
| `…_tidal.csv` | Tidal-bridge blobs: one row per blob and band |
| `…_tidal_pairs.csv` | The interacting pairs |

Fluxes are in nJy (AB zeropoint 31.4). Sizes are half-light radii in arcsec. `x_pix` and `y_pix` are frame pixel
coordinates.

Star rows have positions, `mag_<band>_total` / `flux_<band>_total` for ugrizy, and the parameters that shape their
spikes, halo and saturation (galaxy-only columns are empty for them). Stars have their own random stream, so the
galaxies are identical with or without them; pass `dict(stars=False)` to `build_mock_catalogue` to leave them out.
Their spike angles (shared by every star in an image) and saturated cores are drawn when the images are made.

Galaxy rows also carry `hubble_type` and the parameters that shape its structure (`n_arms`, `arm_pitch_deg`,
`arm_strength`, `arm_sharpness`, `arm_phase_deg`, `arm_start_h`, `barred`, `irregularity`, `structure_seed`), from
their own random stream; pass `dict(hubble_types=False)` to leave them out. Structure is drawn on discs with
`re_disc_arcsec` of at least `IMAGE_CONFIG["structure_min_re_arcsec"]` (0.4").

The added low-surface-brightness galaxies are flagged by `lsb_population` (`extended_dirr`, `udg`, `almost_dark`)
and also carry their drawn central g-band surface brightness `mu0_g` and Sersic index `sersic_n`; they have their own
random stream, so the rest of the catalogue is identical without them (`dict(lsb_galaxies=False)`).

## Generate images

```bash
# every catalogue, the coadds needed for training
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images

# only these catalogues
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --catalogue train test

# save every coadd of every catalogue
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --all-images

# one coadd per catalogue: 10 years, nominal PSF
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --epoch 10y

# alternative to epoch flag. this example generates one coadd with 50 r-band visits, r FWHM 1.3". Other bands scaled proportionally to 10 yr Nexp ratios and expected FWHMs.
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --n-exp 50 --psf-fwhm 1.3
```

For a single coadd, give either `--epoch` (`1m` to `11m` or `1y` to `10y`, at the nominal PSF in every band) or both
`--n-exp` and `--psf-fwhm` (r-band visits and FWHM in arcsec; the other bands scale with LSST's 10-year visit plan and
the nominal PSF ratios).

Each catalogue is rendered once with GalSim (bulge + disc Sersic profiles, Gaussian star-forming regions and tidal
blobs) at a narrow 0.45″ PSF. Coadds are then made from that render for a grid of 7 survey depths (1 month, 6 months, 1,
3, 5, 8 and 10 years of visits) × 6 seeings (r-band PSF FWHM 0.7″ to 2.0″), by broadening it to the target PSF and
adding sky and source noise for that many visits. Depth and seeing vary independently, so a detector cannot learn to
link them.

By default every coadd is listed in each catalogue's manifest and saved, except for `train` and `valid`: the training
code rebuilds those on the fly from the saved base render and manifest, identically every time, so they take no disk
space. At the full 5000 × 5000 pixel size each saved coadd (signal + variance) is 1.2 GB, so for the four standard
catalogues the default run needs about 100 GB, `--all-images` about 200 GB and a single coadd about 7 GB. Saving this partial dataset with a small psf allows for the data to be degraded on the fly. These degraded images can then removed from memory after they have been processed.

Rendering is the slow step; it runs in parallel over `--n-workers` processes (default: up to 8). Outputs go to
`<out-dir>/<catalogue>/`:

| File | Contents |
|---|---|
| `base_clean_signal.npy` | Noise-free render at the 0.45″ base PSF, float32 (band, y, x) in nJy per pixel |
| `base_sfregions_signal.npy`, `base_tidal_signal.npy`, `base_spikes_signal.npy` | The light of the star-forming regions, the tidal features and the stars' diffraction spikes alone, on the same grid: truth for learning where each is |
| `base_meta.json` | Bands, canvas origin in frame pixels, shape, pixel scale, zeropoint, base PSF |
| `coadd_manifest.json` | Every coadd: visits and PSF FWHM per band, noise seed, and whether it was saved |
| `<catalogue>_<coadd>_signal.npy` | A saved coadd (nJy per pixel), e.g. `test_1y_fwhm110_signal.npy`, `test_10y_nominal_signal.npy` or `test_nexp50_fwhm130_signal.npy` |
| `<catalogue>_<coadd>_variance.npy` | Its per-pixel variance |
| `<catalogue>_<coadd>_mask.npy` | Its mask, uint8 (band, y, x). Bit values are in `base_meta.json` under `mask_planes`, named like the LSST stack's planes: `SAT` (1) marks the saturated star cores |
| `<catalogue>_<coadd>_psf.npy` | Its PSF, float32 (band, 41, 41), centred on the middle pixel (`psf_stamp_pix` in `IMAGE_CONFIG`). Cut out, not renormalised: each band sums to the fraction of a point source's light inside the stamp |

Bands are in `ugrizy` order. A catalogue position (`x_pix`, `y_pix`) falls in array pixel
`[:, round(y_pix) - origin_y, round(x_pix) - origin_x]`, with `origin` from `base_meta.json`.

By default the canvas spans the galaxy centres, so its size and origin vary slightly between realisations. Pass
`--canvas-npix N` (usually the catalogues' `--npix`) to make every image exactly frame pixels `[0, N)` on both axes
with origin `(0, 0)`, e.g. to line images up with a fixed patch grid.

The PSF is constant across the image and comes in two models, named by `psf_model` in `base_meta.json` and in each
coadd's `coadd_manifest.json` entry:

- `gaussian` (default): a circular Gaussian in every band, with FWHM per band (arcsec) `fwhm_arcsec`. Stars are not pure
  PSFs: they add a Moffat halo and diffraction spikes (about 5% of their flux) to the Gaussian core, with a shape drawn
  per star.
- `moffat` (`--psf-model moffat`, one coadd only): a Moffat core (`fwhm_arcsec`, `moffat_beta`) plus the stars' halo and
  diffraction spikes, with one shape per band (`psf_wings`), and every source is convolved with all of it. Stars are
  pure PSFs, and galaxies share the halo and spikes. The core FWHMs scale from `moffat_fwhm` in `IMAGE_CONFIG`
  (sharpest in r and i), and each band's FWHM and Moffat index get a small random jitter from the coadd's seed. The
  saved base render is drawn with the core, and the coadd adds the halo and spikes (`psf.py`); `base_spikes_signal.npy`
  then holds the light of the PSF's spikes in the coadd, from every source.

### Large training frames

A detector learns the rare classes (UDGs, extended dIrrs, almost-dark galaxies, BCGs) only if it sees hundreds of
each. Raising their rates in a small frame crowds it (diffuse light everywhere, LSB galaxies overlapping) and teaches a
sky that does not exist, so make the training frame larger instead: at the default rates a 14000 x 14000 px frame
(0.6 deg^2) holds ~480 UDGs, ~480 extended dIrrs, ~240 almost-dark galaxies and ~24 BCGs, with LSB footprints covering
~2% of it. Rendering it at once needs ~19 GB of memory, so it is built as independent parts and stitched:

```bash
python scripts/generate_large_training_frame.py --out-dir ~/mocks_large --parts-per-side 2 --part-npix 7000
```

This writes a `train` split (catalogues and images, coadds rebuilt on the fly) to use with the usual valid, calib and
test splits. Galaxies on a seam lose the light that fell outside their own part, and structure does not continue
across seams; both are negligible for training. Each 7000 px part takes about an hour on a laptop; `--parts` builds
only some of them, so two shells can share the work (`--parts 0 1` and `--parts 2 3`), and the frame is stitched once
all are done. The frame and its component images take 19 GB, and training reads its tiles at random: keep it on a
fast local disk (from a USB drive training ran about ten times slower).

## Truth tables for scoring a detector

A detector that seeds a deblender should find one peak per component that is a local maximum. `truth_peaks.py` builds
the truth tables for that from a catalogue and a coadd's noise-free image:

- `truth_table` lists one row per galaxy, star, star-forming region and tidal blob (`kind`), with its position,
  per-band flux, parent galaxy, `population` and `hubble_type`, and per band whether it is a local maximum of the
  coadd's noise-free image (`peak_<band>`) with that maximum's S/N and offset.
- `find_truth_peaks` returns every local maximum of the noise-free image above a per-pixel S/N floor, per band, with
  the component it belongs to (`truth_id`, or -1 if none is within the match radius).

The noise-free image is the base render broadened to the coadd's PSF, so coadds rebuilt on the fly (train, valid) work
too. Spiral arms, bars and irregular modes are not components: they move a disc's own light around, so a maximum they
make is matched to its galaxy or a star-forming region on it. A galaxy's flux excludes its star-forming regions and
tidal blobs, which have their own rows. At LSST resolution most star-forming regions and all tidal blobs are not
separate maxima, so score completeness against the components with `peak_<band>` set rather than against every row.

## Example commands

Run from the repo folder with the environment active (`source .venv/bin/activate`). The folders under `~/mocks` are
just examples; use any location outside the repo.

**Catalogues**

```bash
# Default: 4 catalogues (train, valid, calib, test), 5000 x 5000 px
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues

# 6 catalogues: train, valid, calib, test, extra1, extra2
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues --n-catalogues 6

# Numbered instead of train/valid/...: mock_catalogue_1, _2, _3
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues --n-catalogues 3 --numbered

# One small, quick catalogue (1000 x 1000 px, 'train')
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues --n-catalogues 1 --npix 1000

# Use the full COSMOS2025 catalogue instead of the shipped subset
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues --cosmos /path/to/cosmos2025_cat.npz
```

**Images: the full depth x seeing grid**

```bash
# Every catalogue; all 42 coadds listed, saved for all but train/valid (rebuilt on the fly in training)
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images

# Save every coadd of every catalogue (~200 GB at full size)
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --all-images

# Only some catalogues
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --catalogue test
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --catalogue calib test
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --catalogue 2
```

**Images: one coadd per catalogue**

```bash
# 10-year depth at the nominal PSF
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --epoch 10y

# Other depths: 1m ... 11m, 1y ... 10y
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --epoch 1y
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --epoch 6m --catalogue test

# Custom depth and seeing: 50 r-band visits at 1.3" r-band FWHM (both flags required)
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --n-exp 50 --psf-fwhm 1.3

# Worst-case seeing at full 10-year depth (184 r-band visits)
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --n-exp 184 --psf-fwhm 2.0

# 1-year depth on a 4200 x 4200 px patch, Moffat PSF with halo and spikes shared by every source
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues --n-catalogues 1 --numbered --npix 4200
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --epoch 1y --psf-model moffat --canvas-npix 4200

# More parallel rendering processes (default: up to 8)
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --n-workers 10
```

**Quick end-to-end test**

```bash
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/test/catalogues --n-catalogues 1 --npix 1000
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/test/catalogues --out-dir ~/mocks/test/images --epoch 10y
```

**Help and tests**

```bash
python scripts/generate_mock_catalogues.py --help
python scripts/generate_mock_images.py --help
pytest -q
```

## Python API

```python
from mock_lsst_image_generation import build_mock_catalogue, load_cosmos2025_catalogue, save_mock_catalogue

cosmos = load_cosmos2025_catalogue()  # shipped subset; or pass a path
cat = build_mock_catalogue(cosmos, dict(seed=7, npix=2000))  # any CONFIG key can be overridden
cat.galaxies, cat.sfregions, cat.tidal_blobs, cat.tidal_pairs
save_mock_catalogue(cat, "my_mock.csv")

from mock_lsst_image_generation import generate_catalogue_images
generate_catalogue_images(cat, "test", "my_images", mode="single", epoch="10y", n_workers=8)  # or "training" / "all"
```

## Tests

```bash
pip install pytest
pytest -q
```

The tests (about a minute) build small catalogues from the shipped subset and check they are sane, reproducible, and
unchanged from the reference realisation pinned in `tests/test_catalogue.py`. `tests/test_images.py` renders a small
catalogue and checks the image keeps the catalogue's flux, the coadd noise matches its variance, and each image mode
and single-coadd option writes the right files. GitHub Actions runs them on every push
(`.github/workflows/tests.yml`). If you change the model on purpose, the reference test will fail: check the new
catalogue, then update `REFERENCE`.

## Licence and data credit

The code is released under the MIT licence (see `LICENSE`). `data/cosmos2025_subset.npz` is a column/row subset of
the public COSMOS2025 catalogue (COSMOS-Web collaboration); please cite the COSMOS2025 catalogue paper if you use it.
