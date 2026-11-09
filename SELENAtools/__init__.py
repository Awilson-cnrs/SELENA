"""
SELENAtools — TEM image analysis toolkit for nanoparticle size distribution
and label image processing.

Public API
----------
Image loading & preprocessing
    load_tem_image
    normalize_image
    apply_gaussian_blur

Segmentation & particle extraction
    adaptive_threshold
    adaptive_threshold_inv
    global_threshold
    cleanup_mask
    extract_particles
    create_overlay

Label image processing
    check_labelling
    bin_grayscale
    bin_labels
    remap_labels

Histogram analysis
    read_hist_files
    fit_gaussian
    plot_histogram
    save_results
    segmentation_info
    save_single_image_results

Utilities
    load_config
    find_dm3_files
    ensure_analysis_dir
"""

from SELENAtools.image import load_tem_image, normalize_image, apply_gaussian_blur
from SELENAtools.segment import (
    adaptive_threshold,
    adaptive_threshold_inv,
    global_threshold,
    cleanup_mask,
    extract_particles,
    create_overlay,
)
from SELENAtools.labels import (
    check_labelling,
    bin_grayscale,
    bin_labels,
    remap_labels,
)
from SELENAtools.histogram import (
    read_hist_files,
    fit_gaussian,
    plot_histogram,
    save_results,
    segmentation_info,
    save_single_image_results,
)
from SELENAtools.utils import load_config, find_dm3_files, ensure_analysis_dir

__all__ = [
    # image.py
    "load_tem_image",
    "normalize_image",
    "apply_gaussian_blur",
    # segment.py
    "adaptive_threshold",
    "adaptive_threshold_inv",
    "global_threshold",
    "cleanup_mask",
    "extract_particles",
    "create_overlay",
    # labels.py
    "check_labelling",
    "bin_grayscale",
    "bin_labels",
    "remap_labels",
    # histogram.py
    "read_hist_files",
    "fit_gaussian",
    "plot_histogram",
    "save_results",
    "segmentation_info",
    "save_single_image_results",
    # utils.py
    "load_config",
    "find_dm3_files",
    "ensure_analysis_dir",
]
