"""
Histogram analysis for nanoparticle size distributions.

Replaces the `TEMana` class from the original notebook.  The `binwidth`
initialization bug is fixed by requiring `binwidth` as an explicit parameter
in every function that needs it.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def gauss_function(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    """Gaussian curve for fitting size distributions."""
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def read_hist_files(folder_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Read and concatenate all .hist files in a folder.

    Parameters
    ----------
    folder_path : str
        Path to the directory containing .hist files.

    Returns
    -------
    df_concat : pd.DataFrame
        Combined DataFrame with a single "Size (nm)" column.
    file_names : list of str
        Basenames of the .hist files that were read.
    """
    df_list: List[pd.DataFrame] = []
    file_names: List[str] = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".hist"):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath, decimal=".", header=[0])
            df_list.append(df)
            file_names.append(filename)

    df_concat = pd.concat(df_list, ignore_index=True)
    return df_concat, file_names


def fit_gaussian(
    sizes: np.ndarray, binwidth: float = 0.5
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Fit a Gaussian curve to a 1-D array of particle sizes.

    Parameters
    ----------
    sizes : np.ndarray
        1-D array of particle sizes in nm.
    binwidth : float
        Bin width for the histogram used in fitting.

    Returns
    -------
    popt : np.ndarray or None
        Optimized parameters [amplitude, mean, sigma].  Returns None if the
        fit fails.
    pcov : np.ndarray or None
        Covariance matrix of the parameters.  Returns None if the fit fails.
    """
    bins = np.arange(sizes.min(), sizes.max() + binwidth, binwidth)
    np_hist = np.histogram(sizes, bins, density=False)
    frequencies = np_hist[0]
    bin_centres = [(np_hist[1][i] + np_hist[1][i + 1]) / 2.0 for i in range(len(np_hist[1]) - 1)]

    mean = sum(bin_centres * frequencies) / sum(frequencies)
    sigma = np.sqrt(sum(frequencies * (bin_centres - mean) ** 2) / sum(frequencies))

    try:
        popt, pcov = curve_fit(
            gauss_function,
            bin_centres,
            frequencies,
            p0=[max(frequencies), mean, sigma],
            maxfev=5000,
        )
        return popt, pcov
    except RuntimeError:
        print("Couldn't fit")
        return None


def plot_histogram(
    sizes: np.ndarray,
    binwidth: float = 0.5,
    fit_gaussian_flag: bool = True,
    title: str = "",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Plot a size distribution histogram with an optional Gaussian overlay.

    Parameters
    ----------
    sizes : np.ndarray
        1-D array of particle sizes in nm.
    binwidth : float
        Bin width for the histogram.
    fit_gaussian_flag : bool
        Whether to overlay a fitted Gaussian curve.
    title : str
        Plot title (sample name, for example).

    Returns
    -------
    popt : np.ndarray or None
        Fitted Gaussian parameters, or None if fitting was skipped/failed.
    pcov : np.ndarray or None
        Covariance matrix, or None.
    """
    bins = np.arange(sizes.min(), sizes.max() + binwidth, binwidth)
    plt.close()
    plt.hist(sizes, bins=bins, label=f"n = {len(sizes)}")

    popt = None
    pcov = None
    if fit_gaussian_flag:
        result = fit_gaussian(sizes, binwidth)
        if result is not None:
            popt, pcov = result
            xs = np.linspace(sizes.min(), sizes.max(), 1000)
            plt.plot(
                xs,
                gauss_function(xs, *popt),
                linestyle="--",
                color="black",
                label=f"Avg size = {popt[1]:.1f} nm\nSigma = {popt[2]:.2f} nm",
            )

    plt.xlabel("NPs size (nm)")
    plt.ylabel("Frequency")
    if title:
        plt.title(title)
    plt.legend()
    return popt, pcov


def save_results(
    folder: str,
    sizes: List[float],
    params: Dict,
    sample_name: str,
    binwidth: float = 0.5,
) -> str:
    """Save analysis results: .hist file, .params file, and histogram plot.

    Parameters
    ----------
    folder : str
        Output directory (will be created if it does not exist).
    sizes : list of float
        Particle sizes in nm.
    params : dict
        Processing parameters (e.g. blur kernel, threshold settings).
    sample_name : str
        Human-readable sample name used for the plot title.
    binwidth : float
        Histogram bin width.

    Returns
    -------
    str
        Path to the saved histogram PDF.
    """
    os.makedirs(folder, exist_ok=True)
    sizes_arr = np.array(sizes)

    # Save .hist file
    hist_path = os.path.join(folder, "result.hist")
    with open(hist_path, "w") as f:
        f.write("Size (nm)\n")
        for size in sizes:
            f.write(f"{size}\n")

    # Save .params file
    params_path = os.path.join(folder, "result.params")
    with open(params_path, "w") as f:
        f.write(json.dumps(params, indent=2))

    # Save histogram plot
    pdf_path = os.path.join(folder, "result.pdf")
    plot_histogram(sizes_arr, binwidth=binwidth, fit_gaussian_flag=True, title=sample_name)
    plt.savefig(pdf_path)
    plt.close()

    return pdf_path


def segmentation_info(
    block_size: int,
    offset: int,
    cutoff_low: float,
    cutoff_high: float,
) -> Dict:
    """Build a dictionary of segmentation processing parameters.

    Parameters
    ----------
    block_size : int
        Block size used in adaptive thresholding.
    offset : int
        Offset subtracted from the mean in adaptive thresholding.
    cutoff_low : float
        Lower size cutoff in nm for filtering particles.
    cutoff_high : float
        Upper size cutoff in nm for filtering particles.

    Returns
    -------
    dict
        Dictionary with keys 'block_size', 'offset', 'cutoff_low', 'cutoff_high'.
    """
    return {
        "block_size": block_size,
        "offset": offset,
        "cutoff_low": cutoff_low,
        "cutoff_high": cutoff_high,
    }


def save_single_image_results(
    parent_folder: str,
    dm3_filename: str,
    sizes: List[float],
    params: Dict,
    overlay_image: np.ndarray,
) -> str:
    """Save results for a single TEM image.

    Creates the `analysis/` subdirectory (if needed) and writes the .hist,
    .params, and overlay image files.

    Parameters
    ----------
    parent_folder : str
        Parent directory containing the .dm3 file.
    dm3_filename : str
        Basename of the .dm3 file (without extension).
    sizes : list of float
        Particle sizes in nm.
    params : dict
        Processing parameters.
    overlay_image : np.ndarray
        RGB overlay image for visualization.

    Returns
    -------
    str
        Path to the saved overlay image.
    """
    analysis_dir = os.path.join(parent_folder, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    base = dm3_filename.split(".dm3")[0]
    save_path = os.path.join(analysis_dir, base)

    # Save overlay image
    overlay_path = f"{save_path}_overlay.png"
    plt.imsave(overlay_path, overlay_image)

    # Save .params file
    params_path = f"{save_path}.params"
    with open(params_path, "w") as f:
        f.write(json.dumps(params))

    # Save .hist file
    hist_path = f"{save_path}.hist"
    with open(hist_path, "w") as f:
        f.write("Size (nm)\n")
        for size in sizes:
            f.write(f"{size}\n")

    return overlay_path
