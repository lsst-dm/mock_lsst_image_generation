"""Mock LSST-like galaxy catalogues and images for training and testing source detection and deblending."""

from .catalogue_pipeline import (MockCatalogue, build_mock_catalogue, generate_mock_catalogues, load_mock_catalogue,
                                 save_mock_catalogue)
from .config import CATALOGUE_STEM, CONFIG, IMAGE_CONFIG, N_CATALOGUES, PHYS, STAR_CONFIG, catalogue_splits
from .cosmos_catalogue import DEFAULT_COSMOS_PATH, load_cosmos2025_catalogue
from .image_pipeline import generate_catalogue_images, generate_mock_images
from .truth_peaks import find_truth_peaks, truth_table

__all__ = ["CATALOGUE_STEM", "CONFIG", "DEFAULT_COSMOS_PATH", "IMAGE_CONFIG", "N_CATALOGUES", "PHYS", "STAR_CONFIG",
           "MockCatalogue", "build_mock_catalogue", "catalogue_splits", "find_truth_peaks", "generate_catalogue_images",
           "generate_mock_catalogues", "generate_mock_images", "load_cosmos2025_catalogue", "load_mock_catalogue",
           "save_mock_catalogue", "truth_table"]
