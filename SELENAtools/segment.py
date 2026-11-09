"""
Segmentation and particle extraction for TEM nanoparticle images.

This module replaces the `TEMimg.adaptBin()` and `TEMimg.histo()` methods from
the original notebook, providing pure functions with explicit inputs/outputs.
The `imgTMP` bug (referencing a module-level variable instead of `self`) is
fixed by using proper local variables throughout.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np


def adaptive_threshold(
    image: np.ndarray, block_size: int = 351, offset: int = 20
) -> np.ndarray:
    """Apply adaptive Gaussian thresholding to separate nanoparticles from substrate.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image.
    block_size : int
        Size of the neighborhood area for adaptive thresholding (must be odd).
    offset : int
        Constant subtracted from the mean or weighted mean.

    Returns
    -------
    np.ndarray
        Binary mask (uint8) where nanoparticles are white (255) and background
        is black (0).
    """
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, offset
    )
    return binary


def extract_particles(
    binary_mask: np.ndarray,
    calibration_nm: float,
    cutoff_low: float = 0.0,
    cutoff_high: float = 10000.0,
) -> Tuple[List[float], List[np.ndarray]]:
    """Extract nanoparticle sizes and contours from a binary mask.

    Parameters
    ----------
    binary_mask : np.ndarray
        2D uint8 binary image (nanoparticles = 255, background = 0).
    calibration_nm : float
        Pixel-to-nanometer calibration scale.
    cutoff_low : float
        Minimum particle size in nm to include.
    cutoff_high : float
        Maximum particle size in nm to include.

    Returns
    -------
    sizes : list of float
        Particle diameters in nm.
    contours : list of np.ndarray
        Contour arrays that passed the size filter and are not touching the
        image border.
    """
    sizes: List[float] = []
    contours: List[np.ndarray] = []
    all_contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    img_h, img_w = binary_mask.shape

    for contour in all_contours:
        area = cv2.contourArea(contour) * calibration_nm * calibration_nm
        size = np.sqrt(4.0 * area / np.pi)
        x, y, w, h = cv2.boundingRect(contour)
        # Skip contours touching the image border
        if x == 0 or y == 0 or x + w == img_w or y + h == img_h:
            continue
        if cutoff_low < size < cutoff_high:
            contours.append(contour)
            sizes.append(size)

    return sizes, contours


def create_overlay(
    image: np.ndarray,
    contours: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw contours on an RGB copy of the original image for visualization.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image.
    contours : list of np.ndarray
        Contour arrays to draw.
    color : tuple of int
        BGR color for contour lines.
    thickness : int
        Line thickness in pixels.

    Returns
    -------
    np.ndarray
        RGB image with contours drawn in the specified color.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    overlay = cv2.drawContours(rgb, contours, -1, color, thickness)
    return overlay


def cleanup_mask(
    binary_mask: np.ndarray,
    min_object_area: int = 100,
    max_hole_area: int = 500,
) -> np.ndarray:
    """Remove speckle noise from a binary mask using morphological operations.

    Applies morphological opening to remove small isolated white pixels (noise),
    then fills small holes inside the remaining objects.

    Parameters
    ----------
    binary_mask : np.ndarray
        2D uint8 binary image (features = 255, background = 0).
    min_object_area : int
        Minimum area in pixels for an object to survive. Smaller connected
        components are removed. Default removes objects smaller than ~10x10 pixels.
    max_hole_area : int
        Maximum hole area in pixels to fill inside objects. Default fills holes
        smaller than ~22x22 pixels.

    Returns
    -------
    np.ndarray
        Cleaned binary mask (uint8).
    """
    mask = binary_mask.copy()

    # Morphological opening: erosion then dilation — removes small isolated pixels
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Morphological closing: dilation then erosion — fills small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Remove connected components smaller than min_object_area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_object_area:
            cleaned[labels == i] = 255

    # Fill small holes inside remaining objects
    # Invert so holes become white blobs on black background
    inverted = cv2.bitwise_not(cleaned)
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )
    for i in range(1, num_labels2):
        area = stats2[i, cv2.CC_STAT_AREA]
        if area <= max_hole_area:
            # Fill this hole by setting it back to the object value
            cleaned[labels2 == i] = 255

    return cleaned


def adaptive_threshold_inv(
    image: np.ndarray, block_size: int = 351, offset: int = 20
) -> np.ndarray:
    """Apply adaptive thresholding where DARKER regions become the foreground.

    Unlike `adaptive_threshold` which uses THRESH_BINARY_INV (dark = foreground),
    this uses THRESH_BINARY (bright = foreground). Use this when your features
    are brighter than the background.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image.
    block_size : int
        Size of the neighborhood area for adaptive thresholding (must be odd).
    offset : int
        Constant subtracted from the mean or weighted mean.

    Returns
    -------
    np.ndarray
        Binary mask (uint8) where features are white (255) and background
        is black (0).
    """
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, offset
    )
    return binary


def global_threshold(
    image: np.ndarray,
    method: int = cv2.THRESH_OTSU,
    invert: bool = True,
) -> np.ndarray:
    """Apply global Otsu or binary thresholding.

    Useful when the image has a clear bimodal histogram (two distinct intensity
    peaks). Otsu automatically finds the optimal threshold value.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image.
    method : int
        OpenCV thresholding method. Default is cv2.THRESH_OTSU.
    invert : bool
        If True, darker regions become foreground (THRESH_BINARY_INV).
        If False, brighter regions become foreground (THRESH_BINARY).

    Returns
    -------
    np.ndarray
        Binary mask (uint8).
    """
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if method == cv2.THRESH_OTSU:
        ret, binary = cv2.threshold(image, 0, 255, thresh_type | cv2.THRESH_OTSU)
    else:
        ret, binary = cv2.threshold(image, 127, 255, thresh_type)
    return binary


def segmentation_info(
    block_size: int, offset: int, cutoff_low: float, cutoff_high: float
) -> Dict:
    """Build a dictionary describing the segmentation parameters used.

    Useful for saving alongside results so the analysis is reproducible.
    """
    return {
        "adaptBin": f"block_size={block_size}, offset={offset}",
        "histo": f"cutoff=({cutoff_low}, {cutoff_high})",
    }
