---
sessionId: session-261106-125529-ahse
---

# Current State & Analysis

### Project Overview

SELENA is a TEM (Transmission Electron Microscopy) image analysis toolkit with two distinct workflows that need unification:

**Project 1 — Nanoparticle Size Distribution** (`2023-06_TEM-distrib_1011_V2.ipynb`)
- `TEMimg` class: loads `.dm3` TEM images via HyperSpy, normalizes intensity (histogram clipping), applies Gaussian blur, adaptive thresholding to separate nanoparticles from substrate, extracts particle sizes from contours, and saves results.
- `TEMana` class: reads saved `.hist` files, concatenates results across images, fits a Gaussian to the size distribution, and plots/saves histograms.
- Uses: `hyperspy`, `cv2` (OpenCV), `numpy`, `pandas`, `matplotlib`, `scipy.optimize`.

**Project 2 — Label Checking & Data Preparation** (`2024-01_check_labelling_V1.ipynb`)
- `checkLabbeling()`: validates that a label image has exactly 3 classes (no unassigned pixels).
- `bin_grayscale_data()`: 2×2 average downsampling for grayscale images.
- `bin_label_data()`: 2×2 mode downsampling for label images.
- Label remapping logic (reassign class IDs for homogeneous labelling).
- Uses: `PIL.Image`, `numpy`, `matplotlib`, `hyperspy`.

### Issues Identified
- **`SELENAtools/` directory is empty** — all logic lives inside notebooks.
- **`TEMimg` class has bugs**: references `imgTMP` instead of `self` in `histo()` and `save()` methods.
- **`TEMana` class has a bug**: `self.binwidth` is used in `fitHisto()` before being set (only set in `saveHisto()`).
- **Duplicate image loading**: both notebooks load TEM images independently; no shared loading logic.
- **No separation of concerns**: image I/O, preprocessing, analysis, and plotting are mixed in classes.
- **`main.py`** is a PyCharm template, unused.
- **`config.json`** exists but is not used by any code.

### Common Themes
Both workflows share:
- Loading TEM images (`.dm3`) via HyperSpy.
- Image normalization (min-max scaling).
- Label/image validation.
- Saving processed images as TIFF.
- Plotting with matplotlib.

# Technical Design

### Architecture: Modules + Notebooks

The reusable logic moves to `SELENAtools/` as Python modules. Notebooks import from the module and focus on interactive exploration, parameter tuning, and visualization.

```
SELENA/
├── SELENAtools/
│   ├── __init__.py          # Public API exports
│   ├── image.py             # TEM image loading, normalization, blur
│   ├── segment.py           # Thresholding, contour extraction, particle sizing
│   ├── labels.py            # Label validation, binning, remapping
│   ├── histogram.py         # Size distribution analysis, Gaussian fit, plotting
│   └── utils.py             # Path helpers, config loading, file I/O
├── notebooks/
│   ├── nanoparticle_size.ipynb       # Replaces 2023-06_TEM-distrib
│   └── check_labelling.ipynb         # Replaces 2024-01_check_labelling
├── config.example.json
├── requirements.yml
└── .gitignore
```

### Module Responsibilities

#### `image.py` — Image Loading & Preprocessing
- `load_tem_image(path)` → returns normalized `numpy.ndarray` + metadata dict (replaces `TEMimg.load()`).
- `normalize_image(data, clip_threshold=4)` → histogram-clipping normalization.
- `apply_gaussian_blur(image, kernel_size=9)` → Gaussian blur wrapper.
- Key decision: rely on HyperSpy for `.dm3` loading and metadata extraction; return plain numpy arrays for downstream processing.

#### `segment.py` — Segmentation & Particle Extraction
- `adaptive_threshold(image, block_size=351, offset=20)` → binary mask of nanoparticles.
- `extract_particles(binary_mask, calibration_nm, cutoff_low=0.0, cutoff_high=10000)` → list of particle sizes and contours.
- `create_overlay(image, contours, color=(0, 255, 0))` → RGB overlay for visualization.
- Replaces `TEMimg.adaptBin()` and `TEMimg.histo()`.

#### `labels.py` — Label Image Processing
- `check_labelling(label_array, expected_classes=3)` → validates label completeness.
- `bin_grayscale(data, factor=2)` → average pooling downsampling.
- `bin_labels(label, factor=2)` → mode pooling downsampling.
- `remap_labels(label_array, mapping={1: 3, 2: 1, 3: 2})` → reassign class IDs.
- Extracted from `2024-01_check_labelling_V1.ipynb`.

#### `histogram.py` — Size Distribution Analysis
- `read_hist_files(folder_path)` → concatenate `.hist` files into a single pandas DataFrame.
- `fit_gaussian(sizes, binwidth=0.5)` → Gaussian fit with `scipy.optimize.curve_fit`.
- `plot_histogram(sizes, binwidth=0.5, fit_gaussian=True, title='')` → histogram with optional Gaussian overlay.
- `save_results(folder_path, sizes, params, sample_name)` → save `.hist`, `.params`, overlay images.
- Replaces `TEMana` class entirely.

#### `utils.py` — Helpers
- `load_config(config_path)` → load `config.json` for data root paths.
- `find_dm3_files(folder, pattern='60kX')` → discover TEM images.
- `ensure_analysis_dir(folder_path)` → create `analysis/` subfolder.

#### `__init__.py` — Public API
- Exports key functions so notebooks can do `from SELENAtools import load_tem_image, adaptive_threshold, ...`.

### Key Design Decisions

1. **Functions over classes**: The `TEMimg` and `TEMana` classes maintain mutable state (`self.steps`, `self.sizes`, etc.) but are used in a linear pipeline. Functions with explicit inputs/outputs are easier to test, compose, and reuse. State tracking (e.g., processing steps) can be done by the caller if needed.

2. **No deep inheritance**: The two workflows are distinct enough (segmentation vs label processing) that a base class would add complexity without reuse. Shared operations (loading, normalization) are utility functions.

3. **HyperSpy for I/O only**: Use HyperSpy to load `.dm3` files and extract metadata/calibration. All image processing uses `numpy`/`scipy`/`skimage` to avoid HyperSpy's signal model overhead for simple 2D operations.

4. **Notebooks remain interactive**: Notebooks will import from `SELENAtools`, tune parameters, and visualize results — but the core logic lives in version-controlled Python modules.

### Data Flow

```
graph LR
    A[.dm3 TEM Image] --> B[load_tem_image]
    B --> C[normalize_image]
    C --> D[apply_gaussian_blur]
    D --> E[adaptive_threshold]
    E --> F[extract_particles]
    F --> G[plot_histogram]
    G --> H[save_results]
    I[_lab.tif Label Image] --> J[check_labelling]
    J --> K[remap_labels]
    K --> L[bin_labels]
    L --> M[Save as TIFF]
```

### Bug Fixes During Refactoring
- `TEMimg.histo()`: `imgTMP.imgBin` → use local variable passed from caller.
- `TEMimg.save()`: `imgTMP.NFO` → use local variable or pass as argument.
- `TEMana.fitHisto()`: require `binwidth` as a parameter instead of relying on `self.binwidth`.

# Delivery Plan

### Phase 1: Create SELENAtools module structure
- Create `SELENAtools/__init__.py` with public API exports.
- Create `SELENAtools/image.py` with `load_tem_image()`, `normalize_image()`, and `apply_gaussian_blur()`.
- Create `SELENAtools/utils.py` with `load_config()`, `find_dm3_files()`, and `ensure_analysis_dir()`.
- Migrate logic from `TEMimg.load()` and `TEMimg.blur()` into these functions.

### Phase 2: Segmentation and particle extraction module
- Create `SELENAtools/segment.py` with `adaptive_threshold()`, `extract_particles()`, and `create_overlay()`.
- Migrate logic from `TEMimg.adaptBin()` and `TEMimg.histo()`.
- Fix the `imgTMP` → proper variable reference bug.

### Phase 3: Label processing module
- Create `SELENAtools/labels.py` with `check_labelling()`, `bin_grayscale()`, `bin_labels()`, and `remap_labels()`.
- Migrate functions from `2024-01_check_labelling_V1.ipynb`.

### Phase 4: Histogram analysis module
- Create `SELENAtools/histogram.py` with `read_hist_files()`, `fit_gaussian()`, `plot_histogram()`, and `save_results()`.
- Migrate logic from `TEMana` class.
- Fix the `binwidth` initialization bug.

### Phase 5: Update notebooks to use the new modules
- Create `notebooks/nanoparticle_size.ipynb` that imports from `SELENAtools` and demonstrates the size distribution workflow.
- Update `notebooks/2024-01_check_labelling_V1.ipynb` to import label functions from `SELENAtools`.
- Remove class definitions from notebooks; keep only interactive exploration cells.

# Delivery Steps

### ✓ Step 1: Create SELENAtools module structure with image loading and utilities
Set up the `SELENAtools/` package with `__init__.py`, `image.py`, and `utils.py` — extracting image loading, normalization, blur, and path helpers from the notebooks.
- Create `SELENAtools/__init__.py` that exports all public functions.
- Create `SELENAtools/image.py` with `load_tem_image(path)`, `normalize_image(data, clip_threshold)`, and `apply_gaussian_blur(image, kernel_size)` — migrated from `TEMimg.load()` and `TEMimg.blur()`.
- Create `SELENAtools/utils.py` with `load_config(path)`, `find_dm3_files(folder, pattern)`, and `ensure_analysis_dir(folder)`.
- Update `requirements.yml` to add `opencv-python` (used by the segmentation logic but currently missing).

### ✓ Step 2: Create segmentation and particle extraction module
Create `SELENAtools/segment.py` with thresholding, contour extraction, and overlay functions — migrated from `TEMimg.adaptBin()` and `TEMimg.histo()`.
- Implement `adaptive_threshold(image, block_size, offset)` returning a binary mask.
- Implement `extract_particles(binary_mask, calibration_nm, cutoff_low, cutoff_high)` returning particle sizes and filtered contours.
- Implement `create_overlay(image, contours, color)` for visualization.
- Fix the `imgTMP` bug where `histo()` and `save()` referenced a module-level variable instead of `self`.

### ✓ Step 3: Create label processing module
Create `SELENAtools/labels.py` with label validation, binning, and remapping — migrated from `2024-01_check_labelling_V1.ipynb`.
- Implement `check_labelling(label_array, expected_classes)` to validate no unassigned pixels.
- Implement `bin_grayscale(data, factor)` and `bin_labels(label, factor)` for 2×2 downsampling.
- Implement `remap_labels(label_array, mapping)` to reassign class IDs for homogeneous labelling.

### ✓ Step 4: Create histogram analysis module
Create `SELENAtools/histogram.py` with size distribution analysis, Gaussian fitting, and plotting — migrated from the `TEMana` class.
- Implement `read_hist_files(folder_path)` to concatenate `.hist` files into a DataFrame.
- Implement `fit_gaussian(sizes, binwidth)` with proper `binwidth` as a parameter (fixing the initialization bug).
- Implement `plot_histogram(sizes, binwidth, fit_gaussian, title)` with optional Gaussian overlay.
- Implement `save_results(folder, sizes, params, sample_name)` to write `.hist`, `.params`, and overlay images.

### ✓ Step 5: Update notebooks to use the new modules
Rewrite the notebooks to import from `SELENAtools` instead of defining classes inline, keeping them as interactive entry points.
- Create `notebooks/nanoparticle_size.ipynb` that demonstrates the full size distribution workflow using `SELENAtools` functions.
- Update `notebooks/2024-01_check_labelling_V1.ipynb` to import label functions from `SELENAtools.labels`.
- Remove `TEMimg` and `TEMana` class definitions from notebooks; replace with function calls and interactive parameter tuning cells.