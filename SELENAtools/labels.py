"""
Label image processing: validation, binning, and remapping.

Extracted from `2024-01_check_labelling_V1.ipynb` for reuse in both notebooks
and scripts.
"""

from typing import Dict

import numpy as np


def check_labelling(label_array: np.ndarray, expected_classes: int = 3) -> str:
    """Validate that a label image contains exactly the expected number of classes.

    Parameters
    ----------
    label_array : np.ndarray
        2D label image (integer dtype).
    expected_classes : int
        Number of unique label values expected (default 3).

    Returns
    -------
    str
        Status message: "No unassigned pixels" on success.

    Raises
    ------
    ValueError
        If the number of unique values does not match `expected_classes`.
    """
    unique_values = np.unique(label_array)
    if len(unique_values) != expected_classes:
        raise ValueError(
            f"Unlabelled pixels or incorrect number of labels: "
            f"expected exactly {expected_classes} unique values, "
            f"got {len(unique_values)}."
        )
    print("No unassigned pixels")
    return "No unassigned pixels"


def bin_grayscale(data: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample a grayscale image by average pooling.

    Parameters
    ----------
    data : np.ndarray
        2D grayscale array.
    factor : int
        Downsampling factor (default 2, halves the resolution).

    Returns
    -------
    np.ndarray
        Downsampled array with the same dtype as input.
    """
    if data.ndim != 2:
        raise ValueError("Input must be a 2D grayscale array.")

    h, w = data.shape
    h_trim, w_trim = h // factor * factor, w // factor * factor
    trimmed = data[:h_trim, :w_trim]
    binned = trimmed.reshape(h_trim // factor, factor, w_trim // factor, factor).mean(
        axis=(1, 3)
    )
    return binned.astype(data.dtype)


def bin_labels(label: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample a label image by mode pooling.

    Parameters
    ----------
    label : np.ndarray
        2D label array (integer dtype).
    factor : int
        Downsampling factor (default 2).

    Returns
    -------
    np.ndarray
        Downsampled label array with the same dtype as input.
    """
    if label.ndim != 2:
        raise ValueError("Input must be a 2D label array.")

    h, w = label.shape
    h_trim, w_trim = h // factor * factor, w // factor * factor
    trimmed = label[:h_trim, :w_trim]
    blocks = trimmed.reshape(h_trim // factor, factor, w_trim // factor, factor)

    mode_values = np.zeros(
        (h_trim // factor, w_trim // factor), dtype=label.dtype
    )
    for i in range(h_trim // factor):
        for j in range(w_trim // factor):
            block = blocks[i, :, j, :]
            mode_values[i, j] = np.bincount(block.ravel()).argmax()

    return mode_values


def remap_labels(
    label_array: np.ndarray, mapping: Dict[int, int]
) -> np.ndarray:
    """Reassign label class IDs for homogeneous labelling.

    Parameters
    ----------
    label_array : np.ndarray
        2D label array.
    mapping : dict
        Mapping from old class ID to new class ID, e.g. {1: 3, 2: 1, 3: 2}.

    Returns
    -------
    np.ndarray
        Copy of the label array with remapped class IDs.
    """
    remapped = np.copy(label_array)
    for old_id, new_id in mapping.items():
        remapped[label_array == old_id] = new_id
    return remapped
