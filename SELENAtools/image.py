"""
Image loading, normalization, and preprocessing for TEM images.

This module replaces the `TEMimg.load()` and `TEMimg.blur()` methods from the
original notebook, providing pure functions that work on plain numpy arrays.
"""

from typing import Dict, Tuple

import cv2
import hyperspy.api as hs
import numpy as np


def load_tem_image(path: str) -> Tuple[np.ndarray, Dict]:
    """Load a .dm3 TEM image via HyperSpy and return normalized data with metadata.

    Parameters
    ----------
    path : str
        Full path to the .dm3 file (may include a glob pattern).

    Returns
    -------
    image : np.ndarray
        Normalized 2D uint8 image.
    metadata : dict
        Dictionary with keys: 'name', 'spec', 'operator', 'microscope', 'calib'.
    """
    imgHS = hs.load(path)
    metadata = _extract_metadata(imgHS)
    image = normalize_image(imgHS.data)
    return image, metadata


def normalize_image(data: np.ndarray, clip_threshold: int = 4) -> np.ndarray:
    """Normalize a 2D image using histogram-clipping then min-max scaling.

    Parameters
    ----------
    data : np.ndarray
        Raw 2D image data (any numeric dtype).
    clip_threshold : int
        Minimum pixel count in a histogram bin before the intensity level is
        considered valid.  Values above the highest valid level are masked out.

    Returns
    -------
    np.ndarray
        Normalized uint8 image in range [0, 255].
    """
    min_val, max_val = np.min(data), np.max(data)
    if max_val == min_val:
        return np.zeros(data.shape, dtype=np.uint8)

    img_temp = np.rint((data - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)

    # Histogram clipping — find the highest intensity bin with enough pixels
    hist = cv2.calcHist([img_temp], [0], None, [256], [1, 256])
    pix_cut = 255
    for i, count in enumerate(reversed(hist.ravel())):
        if count >= clip_threshold:
            pix_cut = 255 - i
            break

    img_temp = np.ma.masked_where(img_temp > pix_cut, img_temp)
    max_temp = np.max(img_temp)
    if max_temp == 0:
        return np.zeros(img_temp.shape, dtype=np.uint8)
    img_temp = (img_temp * 255.0 / max_temp).astype(np.uint8)
    return img_temp


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Apply Gaussian blur to a 2D grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 image.
    kernel_size : int
        Size of the Gaussian kernel (must be odd).

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def _safe_get(d, key, default=""):
    """Safely access a key in a HyperSpy DictionaryTreeBrowser or dict.

    DictionaryTreeBrowser supports ``[]`` bracket access but not ``.get()``.
    """
    try:
        return d[key]
    except (KeyError, AttributeError, TypeError):
        return default


def _extract_metadata(imgHS) -> Dict:
    """Extract useful metadata from a HyperSpy signal loaded from a .dm3 file.

    Returns a dict with keys: 'name', 'spec', 'operator', 'microscope', 'calib'.
    """
    md = imgHS.original_metadata
    tg = md["ImageList"]["TagGroup0"]
    image_tags = tg["ImageTags"]

    # Session Info may not be present in all .dm3 files
    session = {}
    try:
        session = image_tags["Session Info"]
    except (KeyError, AttributeError):
        pass

    # Calibration scale (nanometers per pixel)
    calibrations = tg["ImageData"]["Calibrations"]["Dimension"]["TagGroup0"]
    calib = _safe_get(calibrations, "Scale", 1.0)

    return {
        "name": _safe_get(tg, "Name"),
        "spec": _safe_get(session, "Specimen"),
        "operator": _safe_get(session, "Operator"),
        "microscope": _safe_get(session, "Microscope"),
        "calib": calib,
    }
