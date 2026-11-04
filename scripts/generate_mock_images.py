"""Render mock catalogues into LSST-like coadd images (run generate_mock_catalogues.py first).

Each catalogue is rendered once with GalSim; coadds at every survey depth (1 month to 10 years) and seeing (r-band
FWHM 0.7" to 2.0") are then made from that render. By default every catalogue in --catalogue-dir is processed, all
coadds are listed in each manifest, and all are saved except train/valid's: training code rebuilds those on the fly
from the saved base render, so they take no disk space.

    # all catalogues, coadds needed for training
    python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images
    # only these catalogues
    python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --catalogue train test
    # save every coadd of every catalogue
    python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --all-images
    # one coadd: 10 years of visits, nominal PSF
    python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --epoch 10y
    # one coadd: 50 r-band visits, r FWHM 1.3"
    python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --n-exp 50 --psf-fwhm 1.3
    # one coadd: 1 year of visits, a Moffat PSF with halo and diffraction spikes for every source, 4200 px patch
    python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --epoch 1y --psf-model moffat --canvas-npix 4200

Keep both folders outside the repo. Outputs go to <out-dir>/<catalogue>/; see
mock_lsst_image_generation/image_pipeline.py for the file list.
"""

import argparse
import os
from pathlib import Path

from mock_lsst_image_generation import generate_mock_images
from mock_lsst_image_generation.coadd_synthesis import EPOCH_CHOICES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalogue-dir", type=Path, required=True,
                        help="where generate_mock_catalogues.py wrote the catalogues")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="output directory; keep it outside the repo, images are large (up to ~200 GB)")
    parser.add_argument("--catalogue", nargs="+", metavar="NAME",
                        help="only these catalogues, e.g. train or 1 (default: all in --catalogue-dir)")
    parser.add_argument("--all-images", action="store_true", help="save every coadd of every catalogue")
    single = parser.add_argument_group("one coadd per catalogue instead of the grid (give --epoch, or --n-exp and "
                                       "--psf-fwhm)")
    single.add_argument("--epoch", choices=EPOCH_CHOICES, metavar="EPOCH",
                        help=f"survey depth at the nominal PSF: {', '.join(EPOCH_CHOICES)}")
    single.add_argument("--n-exp", type=int, help="number of r-band visits (other bands scale with the survey plan)")
    single.add_argument("--psf-fwhm", type=float,
                        help="r-band PSF FWHM in arcsec (other bands scale with the nominal ratios)")
    single.add_argument("--psf-model", choices=["gaussian", "moffat"], default="gaussian",
                        help="gaussian (default), or a Moffat core whose halo and diffraction spikes every source "
                             "shares (one coadd only; see IMAGE_CONFIG moffat_fwhm, moffat_beta, psf_wings)")
    parser.add_argument("--canvas-npix", type=int, metavar="N",
                        help="make every image exactly frame pixels [0, N) on both axes, e.g. the --npix the "
                             "catalogues were built with (default: the canvas spans the galaxy centres)")
    parser.add_argument("--n-workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="parallel rendering processes (default: %(default)s)")
    args = parser.parse_args()
    if (args.n_exp is None) != (args.psf_fwhm is None):
        parser.error("--n-exp and --psf-fwhm must be given together")
    one_coadd = args.epoch is not None or args.n_exp is not None
    if args.epoch is not None and args.n_exp is not None:
        parser.error("give --epoch, or --n-exp with --psf-fwhm, not both")
    if one_coadd and args.all_images:
        parser.error("--all-images makes the whole grid; it cannot be combined with --epoch / --n-exp")
    if args.psf_model == "moffat" and not one_coadd:
        parser.error("--psf-model moffat makes one coadd: give --epoch, or --n-exp with --psf-fwhm")
    mode = "single" if one_coadd else "all" if args.all_images else "training"
    cfg = dict(psf_model=args.psf_model, **(dict(canvas_npix=args.canvas_npix) if args.canvas_npix else {}))
    generate_mock_images(args.catalogue_dir, args.out_dir, args.catalogue, mode, cfg=cfg, n_workers=args.n_workers,
                         epoch=args.epoch, n_exp=args.n_exp, psf_fwhm=args.psf_fwhm)


if __name__ == "__main__":
    main()
