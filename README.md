# ⚡ EMG Processing & Visualization Pipeline for Exoskeleton Evaluation

> **An automated data engineering and biomechanical analysis pipeline to process raw electromyography (EMG) signals and evaluate human muscle activation under varying exoskeleton assistance profiles.**

## 📌 Overview

Evaluating the efficacy of robotic hip exoskeletons requires understanding how different assistance parameters (torque **magnitude** and actuation **delay**) impact human muscle activation. 

This repository contains a complete pipeline that ingests raw ground reaction force (GRF) and EMG data, filters and synchronizes the signals to the human gait cycle, and generates advanced visualizations (Heatmaps, Contour Maps, and Full-Grid Sweeps) to identify the optimal assistance profiles that minimize metabolic cost.

### 🎯 Key Features
* **Automated Batch Processing:** Iterates through multiple subjects, walking speeds, and conditions (Level Ground [LG], Ramp Ascent [RA], Ramp Descent [RD]).
* **Signal Processing:** Implements precise Butterworth filtering, global normalization against baseline walking ("NoExo"), and gait-cycle phase alignment (0-100\%).
* **Muscle-Specific Phase Extraction:** Automatically isolates peak activations during critical biomechanical phases (e.g., Rectus Femoris during 60-100\% swing phase, Bicep Femoris wrap-around 85-10\%).
* **Optimization Analysis:** Calculates `% Change from NoExo` to explicitly pinpoint the optimal and worst torque magnitude/delay combinations.

---

## 📂 Expected Data Structure
**Note: Actual subject data is excluded from this repository for privacy and size constraints.** To use these scripts, you must create a local `data/` directory at the root of the project and structure your raw Vicon/EMG files exactly as follows:

```text
data/
└── raw_data/
    ├── Subject_01/
    │   ├── LG/ (Level Ground)
    │   │   ├── EMG/
    │   │   │   ├── EMG_NoExo.csv
    │   │   │   ├── EMG_NoAssi.csv
    │   │   │   ├── EMG_10p_150ms.csv
    │   │   │   └── ... (27 files total)
    │   │   └── GRF/
    │   │       ├── GRF_NoExo.csv
    │   │       ├── GRF_NoAssi.csv
    │   │       ├── GRF_10p_150ms.csv
    │   │       └── ... (27 files total)
    │   ├── RA/ (Ramp Ascent)
    │   │   ├── EMG/
    │   │   └── GRF/
    │   └── RD/ (Ramp Descent)
    │       ├── EMG/
    │       └── GRF/
    └── Subject_02/
        └── ...
```

_Inside each condition folder (`LG`, `RA`, `RD`), the `EMG` and `GRF` directories should each contain exactly 27 trial files: 25 assistance profiles (5 magnitudes x 5 delays), 1 `NoAssi` baseline, and 1 `NoExo` baseline._

## 🛠️ System Architecture

### 1. Data Processing (`src/processing/`)
* `processing_functions.py`: The core engine. Handles Vicon GRF synchronization, finding heel-strike indices, calculating scaling factors, and applying filtering.
* `emg_batch_process.py`: Traverses nested directories of raw trial data, applies global normalization, and outputs structured, averaged EMG envelopes.

### 2. Visualization Suite (`src/visualization/`)
The pipeline outputs publication-quality figures across several dimensions:
* **Single-Muscle Utilities:** (`emg_plot_single.py`) A lightweight, modular plotting tool to generate a clear, isolated view of a single muscle's mean activation and standard deviation envelope.
* **Envelopes & Sweeps:** (`emg_test_plot.py`, `plot_fullgrid_emg_sweeps.py`) Plot the mean EMG activation (± standard deviation) over the 100% gait cycle, juxtaposing assistance conditions directly against `NoExo` and `NoAssi` baselines.
* **T-Grid Sweeps:** (`plot_Tgrid_emg_sweeps.py`) Generates a structured $2x4$ T-grid layout comparing independent magnitude sweeps and delay sweeps side-by-side.
* **Heatmaps:** (`peak_activation_heatmap.py`, `plot_average_heatmap.py`) Visualizes peak activation reduction/increases on a 2D grid of Magnitude vs. Delay. Includes both individual and N-subject averaged maps using unified global scales.
* **Contour Maps:** (`emg_contour_analysis.py`, `emg_contour_averages.py`) Uses Radial Basis Function (Rbf) interpolation to map the continuous activation space, identifying exact local minima (optimal assistance) and maxima (worst assistance).
* **Summary Scatters:** (`emg_summary.py`) High-level cross-condition comparison mapping overground/treadmill variables.

---

## 📊 Sample Visualizations

*(Note: Add your exported images to `docs/figures/` and uncomment these links)*

---

## 🚀 How to Run (Quickstart)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Process Raw Data
Ensure your local `data/raw/` directory is populated according to the structure outlined above.
```bash
python src/processing/emg_batch_process.py
```
### 3. Generate Visualizations
Once the data is processed into `data/processed/`, you can generate the analytical plots. For example, to generate the aggregated contour maps:
```bash
python src/visualization/emg_contour_averages.py
```

----

## 📁 Repository Structure
* `src/`: Source code for processing and visualization.
* `docs/`: Project presentation slides and exported figures.
* `data/`: _(Gitignored)_ Expected location for local raw and processed datasets.
