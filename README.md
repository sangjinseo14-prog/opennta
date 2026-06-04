# OpenNTA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20518443.svg)](https://doi.org/10.5281/zenodo.20518443)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-brightgreen.svg)](https://riverbankcomputing.com/software/pyqt/)

OpenNTA is a desktop application for **Nanoparticle Tracking Analysis (NTA)**.
It integrates the standard NTA components — TIFF preprocessing, Fiji/TrackMate
detection, drift correction, MSD fitting, and size-distribution estimation —
behind a single PyQt5 window, giving the user full control over each step of
the analysis.

## Table of contents

- [What it does](#what-it-does)
- [Installation](#installation)
  - [Requirements](#1-requirements)
  - [Set up a virtual environment](#2-set-up-a-virtual-environment)
  - [Install the package](#3-install-the-package)
  - [Install U-Net bundle assets (optional)](#4-install-u-net-bundle-assets-optional)
- [How to run](#how-to-run)
- [How to use](#how-to-use)
  - [Folder flow (Track → Batch → Analysis)](#folder-flow-track--batch--analysis)
  - [Track tab](#track-tab)
  - [Analysis tab](#analysis-tab)
  - [Batch tab](#batch-tab)
  - [Outputs](#outputs)
- [Development](#development)
  - [Project layout](#project-layout)
  - [Testing](#testing)
  - [Logging](#logging)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Releases and tags](#releases-and-tags)
- [License](#license)
- [Contact](#contact)

## What it does

**The standard NTA workflow** is implemented end-to-end:

1. *Detect & track* — particles across a TIFF stack. TrackMate handles
   detection; OpenNTA automates its launch and parameter passing.
2. *Correct drift* — bulk flow in the chamber would otherwise be picked up as
   "diffusion." OpenNTA ships several strategies, from a simple mean
   subtraction to a spatially-varying numerical field, plus an experimental
   U-Net based field if you have the trained weights.
3. *Compute MSD* — per track, and estimate diffusion coefficients.
4. *Convert D → diameter* — via Stokes–Einstein. Two additional implementations
   are included: parametric FTLA (Saveyn 2010) and non-parametric iterative
   (Walker 2012).
5. *Plot, export, repeat* — either one file at a time in the Analysis tab, or
   in bulk through the Batch tab.

**OpenNTA** extends the standard NTA workflow with:

1. *Automated thresholding* — the threshold is determined in a statistically
   principled way, so manual tuning is no longer required for each measurement.
2. *Extensible architecture* — dependencies are layered to avoid circular
   imports, and correction/histogram algorithms can be registered with
   decorators.
3. *Hardware independence* — home-built NTA setups can be used with proper
   calibration and parameter tuning, and hardware innovations can be
   incorporated by adapting the tracking module.
4. *Parameter accessibility* — the UI exposes the key parameters directly, and
   additional parameters can be adjusted in the source code as needed.

## Installation

There are two installation paths that share the same prerequisites and
virtual-environment setup:

- **Developers** — clone the repository and install editable (`pip install -e .`)
  so source edits and tests pick up immediately.
- **Non-developers** — download the prebuilt wheel from the latest GitHub
  Release and install it.

Pick the path you need under [Install the package](#3-install-the-package); the
rest of this section applies to both.

### 1. Requirements

- Python **3.10+**
- A working **Fiji/ImageJ** install with the **TrackMate** plugin (needed for
  the Track tab; not required to import the package or run the analysis-only
  workflow).

### 2. Set up a virtual environment

For the **developer** path, clone first:
> ```sh
> git clone https://github.com/sangjinseo14-prog/opennta.git
> cd opennta
> ```

For the **non-developer** path, create a working folder:
> ```sh
> cd ~
> mkdir opennta && cd opennta
> ```

Then, in either path, create and activate the venv:

```sh
python -m venv .venv
```
```sh
# For PowerShell(windows)
.venv\Scripts\Activate.ps1

# For CMD
.venv\Scripts\activate.bat

# For macOS/Linux             
source .venv/bin/activate
```             
```sh
python -m pip install --upgrade pip
```

All install commands below assume the venv is active and you are in the
repository root (developer path) or the working folder (non-developer path).

### 3. Install the package

#### Developer path (editable install)
> **a) core:** Tracking / Analysis / Batch workflow.
> ```sh
> python -m pip install -e .
> ```
>
> **b) unet:** add U-Net drift-correction in Analysis
> (pulls in TensorFlow and OpenCV).
> ```sh
> python -m pip install -e ".[unet]"
> ```
>
> **c) dev:** editable install with the test extras (recommended).
> ```sh
> python -m pip install -e ".[dev,full]"
> ```

#### Non-developer path (wheel from GitHub Release)
> Each [GitHub Release](https://github.com/sangjinseo14-prog/opennta/releases/latest)
> attaches a wheel. Place the downloaded wheel file in the working folder
> (`~/opennta/*.whl`), then install:
> 
> **a) core:** Tracking / Analysis / Batch workflow.
> ```sh
> python -m pip install opennta-<version>-py3-none-any.whl
>```
> **b) unet:** add U-Net drift-correction in Analysis (recommended)
> ```sh
> python -m pip install "opennta-<version>-py3-none-any.whl[unet]"       
> ```

#### Verify the install
```sh
python -c "import opennta; print(opennta.__version__)"
```

### 4. Install U-Net bundle assets (optional)

Only needed if you installed the `[unet]` extra and intend to use the U-Net
drift-correction mode. The bundle assets (`best.weights.h5`, `meta.csv`,
`norm.json`, `reference_field.npz`) are not tracked in git and are fetched as
a single `unet_assets.zip` from the GitHub Release.

#### Developer path (download script)

For most users, `#Default` is sufficient.
```sh
#Default
python scripts/download_assets.py

#Specific version
python scripts/download_assets.py --tag assets_v1.0.0

#Force re-download
python scripts/download_assets.py --force               
```

The files are extracted into
`src/opennta/analysis/unet_field/unet_bundle/unet_assets/`. If every expected file
is already present the download is skipped; otherwise the zip is fetched, its
SHA-256 verified, and the contents extracted.

#### Non-developer path (manual)

Download `unet_assets.zip` from the matching `assets_v*` GitHub release and
extract its contents into the installed bundle's `unet_assets/` directory — print
the exact target with:

```sh
python -c "import opennta.analysis.unet_field.unet_bundle as b, pathlib; print(pathlib.Path(b.__file__).parent / 'unet_assets')"
```

Once the assets are in place the U-Net mode is ready to use.

## How to run

After installation, either of the two forms below launches the GUI:

```sh
opennta                              # console-script entry point (recommended)
python -m opennta                    # equivalent
```

## How to use

### Folder flow (Track → Batch → Analysis)

TIFF files should be organized as shown below:

```text
Folder/                                          <- Select this folder in the Track tab
├── NTA_1_<name>/                                # sample name
│   ├── 1/                                       # measurement 1
│   ├── ...
│   └── n/                                       # measurement n
├── ...
├── NTA_m_<name>/
```

After running the Track tab, OpenNTA generates an output folder under `Folder/`:

```text
└── OpenNTA_Results_YYYYMMDD_HHMMSS/             <- Select this folder in the Batch tab
    ├── NTA_1_<name>/
    │   ├── *_first_frame_quality.csv
    │   ├── *_first_frame_Cheng-Schwartzman_fit.csv
    │   ├── *_first_frame_Cheng-Schwartzman_fit.png
    │   ├── *_spots.csv                          <- Select one of these in the Analysis tab
    │   └── ...                                  # repeated for measurements 1 to n
    ├── ...
    ├── NTA_m_<name>/
```

After running the Batch tab, OpenNTA generates an output folder under
`Folder/OpenNTA_Results_YYYYMMDD_HHMMSS/`:

```text
    └── size_YYYYMMDD_#/
        ├── individual_results/                  # non-aggregated, always-on outputs
        │   ├── NTA_1_<name>/
        │   │   ├── NTA_1_<name>_1_diameter.csv  # diameter values
        │   │   ├── NTA_1_<name>_1_report.html   # per-file statistical report
        │   │   ├── ...
        │   │   ├── NTA_1_<name>_n_diameter.csv
        │   │   └── NTA_1_<name>_n_report.html
        │   ├── ...
        │   └── NTA_m_<name>/
        ├── merged_results/                      # aggregated results
        │   ├── merged_group_1_diameter.csv      # per-group merged diameters
        │   ├── merged_group_1_files.txt         # column → source-file mapping
        │   ├── merged_group_1_report.html       # per-group statistical report
        │   │   ...
        │   ├── merged_group_m_diameter.csv
        │   ├── merged_group_m_files.txt
        │   └── merged_group_m_report.html
        ├── velocity_fields/                     # only when a Field corrector is used
        │   ├── NTA_1_<name>/
        │   │   ├── NTA_1_<name>_1_drift_field.png   # Numerical corrector output
        │   │   └── NTA_1_<name>_1_ml_field.png      # U-Net corrector output
        │   └── ...
        ├── corrected_tracks/                    # only when "csv" (corrected tracks) is checked
        │   ├── NTA_1_<name>/
        │   │   └── NTA_1_<name>_1_corrected.csv     # drift-corrected particle positions
        │   └── ...
        ├── field_csv/                           # only when "Export velocity fields as csv" is checked
        │   ├── NTA_1_<name>/
        │   │   └── NTA_1_<name>_1_field.csv         # flow field x,y,u,v (u,v in m/s)
        │   └── ...
        └── batch_analysis_log.txt               # processing log for the batch run
                                                 # (acquisition config, analysis settings, drift-corrector params)
```

### Track tab

1. The app tries to auto-detect Fiji on first launch (see
   `TabTrackingHelper.find_fiji_in_common_places()` for preset paths). If it
   cannot find Fiji, point at the executable in the Track tab.
2. **Find** the root `Folder` and select the subfolders to be analyzed
   (starting with `NTA_*`).
3. Pick a **tracking method** under the *Tracking method* group box:
   - *Combined* — choose a single end-to-end pipeline from the **Method**
     combo box (TrackMate is registered by default).
   - *Split* — pair an independent **Detector** with an independent
     **Linker**. Useful when you want to mix and match algorithms registered
     separately (see *Adding a new tracking method* below).
4. Hit **Run tracking**. If the selected method needs configuration (TrackMate
   does), an interactive configuration dialog opens (see
   *Tracking method 1: TrackMate* below). Confirm the parameters to start the
   run.
5. OpenNTA creates `OpenNTA_Results_YYYYMMDD_HHMMSS/` and organizes the
   tracking artifacts into per-sample subfolders.

<p align="center">
<img width="600" alt="tracking" src="https://github.com/user-attachments/assets/d542c26c-b7e8-44c2-86f7-994322967b0a" />
</p>

#### Tracking method 1: TrackMate

The TrackMate configuration dialog provides a live detection preview, a
quality histogram, automated threshold fitting, and adjustable linker
parameters. It is launched automatically when **Run tracking** is hit with
TrackMate selected as the method.

<p align="center">
<img width="600" alt="trackmate" src="https://github.com/user-attachments/assets/28aaaad2-76a0-46f3-b9aa-e82342c9d868" />
</p>


### Analysis tab

1. Load a `_spots.csv` produced by the Track tab (or any TrackMate run with
   the same column layout).
2. Open and adjust the configuration settings (sensor size, magnification,
   fps, temperature, viscosity).

   <p align="center">
   <img width="300"alt="config" src="https://github.com/user-attachments/assets/9fb7908e-701b-443d-8c9a-5c8f2d315df1" />
   </p>



   **Units.** Each field is entered in the unit shown next to it in the dialog:

   | Field | Unit | Default |
   |-------|------|---------|
   | Sensor pixel size | µm | 6.5 |
   | Lens magnification | × | 20 |
   | Frames per second | Hz | 25 |
   | Temperature | **K** (Kelvin) | 295.15 |
   | Viscosity | **mPa·s** | 0.9544 |

   The unit is stored alongside each value in `config.json`.

   <details>
   <summary><b>Config file location</b></summary>

   User configuration is stored as `config.json` under the platform's standard
   user-config directory (resolved via `platformdirs.user_config_dir`). The
   last-used Fiji path is stored next to it as `fiji_path.txt`. Each value is
   stored next to its unit, e.g. `"eta": {"value": 0.89, "unit": "mPa·s"}`.

   | OS | Path |
   |----|------|
   | Windows | `%APPDATA%\opennta\temp\config.json` |
   | macOS | `~/Library/Application Support/opennta/temp/config.json` |
   | Linux | `~/.config/opennta/temp/config.json` (or `$XDG_CONFIG_HOME/opennta/temp/config.json`) |

   </details>

3. Pick a **correction mode** (see *Correction modes* below):
   - *None* — leave positions as-is.
   - *Mean* — subtract the global mean drift.
   - *Field: Numerical* — fit a spatially-varying drift field from the data.
   - *Field: U-Net* — same idea, but the field comes from a pretrained
     network (requires the inference bundle).
4. Hit **Analysis**.
5. Inspect the size distribution, replot with different histogram modes and
   styles, and export the results.

<p align="center">
<img width="600" alt="analysis" src="https://github.com/user-attachments/assets/284a9886-9b83-4949-99bd-52c9abd092ee" />
</p>


#### Correction mode 1 — Global: Mean

Subtracts a single mean displacement vector from every track. Fast, and
appropriate when the bulk flow in the chamber is uniform across the field of
view.

#### Correction mode 2 — Field: Numerical

Fits a spatially-varying drift field directly from the tracked data. Useful
when the flow exhibits clear spatial structure (for example, near the chamber
walls) that a single mean cannot capture.

The dialog also offers an **Interpolation** group box: enable it and set a
**nodes** count to resample the window field onto a finer node grid (bilinear),
or leave it off to use one value per window. Tick **Export velocity fields as
csv** to additionally write the fitted field to the Desktop as
`<spots-name>_<mode>_uv field.csv` (`x, y, u, v`, with `u, v` in m/s).

<p align="center">
<img width="600" alt="numerical" src="https://github.com/user-attachments/assets/ac645c34-049c-458b-bd20-738cd4c30cc6" />
</p>



#### Correction mode 3 — Field: U-Net

Predicts the drift field with a pretrained U-Net. Same idea as the numerical
field, but the field is inferred from a network trained on representative data
rather than fitted on the fly. Requires the U-Net inference bundle (see
*Install U-Net bundle assets*). Like the numerical corrector, its dialog
exposes an **Export velocity fields as csv** checkbox to write the predicted
field as a `*_field.csv`.

<p align="center">
<img width="600" alt="unet" src="https://github.com/user-attachments/assets/39891d3f-53a4-4918-9bb7-030fc6453061" />
</p>



### Batch tab

Process multiple files with a shared configuration. Outputs include per-file
reports and aggregated results. Use the Analysis tab for preliminary
inspection and the Batch tab for handling multiple results at once.

1. **Find** the result folder `~/OpenNTA_Results_YYYYMMDD_HHMMSS/` and select
   the results to be analyzed.
2. Organize groups for aggregating the output (the default grouping is
   sufficient in most cases).
3. Open and adjust the configuration settings. Optionally tick the **csv**
   checkbox to also write drift-corrected track CSVs (into `corrected_tracks/`);
   a Field corrector's **Export velocity fields as csv** option likewise writes
   the per-file flow field into `field_csv/`.
4. Hit **Analysis**.
5. Check the merged report and the per-file outputs in the generated
   `size_YYYYMMDD_#/` folder.

<p align="center">
<img width="600" alt="batch" src="https://github.com/user-attachments/assets/98b4046a-b91e-4775-ae4a-53f29e2fc449" />
</p>

### Outputs

**Tracking output** (under `OpenNTA_Results_{timestamp}/<sample>/`):

- `OpenNTA_Results_{timestamp}/` — main output folder for a tracking run.
- `*_first_frame_quality.csv` — TrackMate spot quality values for the first frame.
- `*_first_frame_Cheng-Schwartzman_fit.csv` — Cheng-Schwartzman fitted curve, point by point.
- `*_first_frame_Cheng-Schwartzman_fit.png` — Cheng-Schwartzman fitted curve plot.
- `*_spots.csv` — tracking result (frame, x, y, track_id, quality).

**Batch output** (under `OpenNTA_Results_{timestamp}/size_{date}_{number}/`):

- `size_{date}_{number}/` — main output folder for a batch analysis run.
- `individual_results/` — per-file, always-on outputs (one subfolder per sample);
  holds only `*_diameter.csv` and `*_report.html`.
- `individual_results/<sample>/*_diameter.csv` — per-file diameter values.
- `individual_results/<sample>/*_report.html` — per-file statistical report.
- `merged_results/merged_group_{N}_diameter.csv` — per-group merged diameters.
- `merged_results/merged_group_{N}_files.txt` — column → source-file mapping
  and per-column particle counts.
- `merged_results/merged_group_{N}_report.html` — per-group statistical report
  (now includes a compact per-file diameter summary under the merged statistics).
- `velocity_fields/<sample>/*_drift_field.png` or `*_ml_field.png` —
  flow-field diagnostics, written only when a Field corrector is active.
- `corrected_tracks/<sample>/*_corrected.csv` — drift-corrected particle
  positions, written only when the batch **csv** checkbox is enabled.
- `field_csv/<sample>/*_field.csv` — the flow field used for correction
  (`x, y, u, v`; `u, v` in m/s), written only when **Export velocity fields as
  csv** is enabled on a Field corrector.
- `batch_analysis_log.txt` — processing log for the batch run, recording the
  acquisition config, analysis settings, and drift-corrector parameters.

## Development

### Project layout

The repository follows the PyPA-recommended **src-layout**: the importable
package lives under `src/opennta/`, tests live at the repository root, and
non-installable helpers (asset downloader, dependency visualiser, launcher)
sit under `scripts/`.

```
opennta/                                     # repository root
├── pyproject.toml                           # build / dependency / tool config (single source)
├── README.md
├── LICENSE
│
├── docs/                                    # style guides & contribution docs
│   ├── CONTRIBUTING.md
│   ├── NAMING_STYLE.md
│   └── COMMENT_STYLE.md
│
├── scripts/                                 # non-installable developer helpers
│   ├── download_assets.py                   # fetches the U-Net bundle assets zip from a GitHub Release
│   └── dependency_drawer.py                 # AST-based import-graph SVG generator
│
├── tests/                                   # pytest suite (synthesised inputs, no sample TIFFs)
│   ├── conftest.py                          # shared session fixtures + Qt stubs
│   ├── _helpers/                            # synth-data generators + registry factories
│   │   ├── synth.py
│   │   └── factories.py
│   ├── unit/                                # one test file per source class, mirrors src tree
│   │   ├── analysis/
│   │   │   ├── test_data_preprocessor.py
│   │   │   ├── test_drift_analyzer.py
│   │   │   ├── test_drift_corrector.py
│   │   │   ├── test_msd_calculator.py
│   │   │   ├── test_diffusion_estimator.py
│   │   │   ├── test_size_distributor.py
│   │   │   ├── mle/                         # test_likelihood.py, test_families.py, ...
│   │   │   └── numerical_field/
│   │   ├── common/
│   │   │   └── test_progress_emitter.py
│   │   └── tracking/
│   │       ├── test_tracking_registries.py
│   │       └── trackmate/
│   ├── integration/                         # end-to-end pipeline tests
│   │   └── test_analysis_pipeline.py
│   └── contracts/                           # registry / skip-list invariants
│       └── test_registry_meta.py
│
└── src/opennta/                             # the installable package (`import opennta`)
    ├── __init__.py                          # version + Windows TF/PyQt DLL preload, subpackage imports
    ├── __main__.py                          # entry point: bootstraps QApplication and launches OpenNtaMainWindow
    │
    ├── analysis/                            # everything downstream of tracking
    │   ├── __init__.py                      # imports plugin subpackages so their @register_* run
    │   ├── analysis_processor.py            # orchestrates the analysis pipeline
    │   ├── types.py                         # AnalysisConfig / AnalysisResults dataclasses
    │   ├── data_preprocessor.py             # CSV loading + cleanup
    │   ├── diffusion_estimator.py           # D → diameter conversion (Stokes–Einstein)
    │   ├── drift_analyzer.py                # drift summary stats
    │   ├── drift_corrector.py               # base class + registry, None / Mean correctors
    │   ├── corrected_track_exporter.py      # Qt-free corrected-track CSV writer (shared by both tabs)
    │   ├── msd_calculator.py                # computes MSD curves from tracks
    │   ├── result_filters.py                # R² / quality filtering shared across tabs
    │   ├── size_distributor.py              # base class + registry, Direct distributor
    │   ├── status_codes.py                  # enum codes used by progress emission
    │   │
    │   ├── numerical_field/                 # data-driven spatial drift correction
    │   │   ├── numerical_corrector.py       # @register_corrector("Field: Numerical")
    │   │   ├── velocity_field.py            # field representation
    │   │   ├── field_sampler.py             # samples drift from raw track data
    │   │   ├── field_smoother.py            # smoothing/regularization of the field
    │   │   ├── node_field.py                # bilinear interpolation onto a node grid
    │   │   └── types.py                     # dataclasses shared by the subpackage
    │   │                                    # (configuration dialog lives in application/tab_analysis/dialogs/numerical/)
    │   │
    │   ├── unet_field/                      # U-Net-based drift correction (opt-in)
    │   │   ├── unet_corrector.py            # @register_corrector("Field: U-Net")
    │   │   ├── field_sampler.py             # samples drift for U-Net inference inputs
    │   │   ├── unit_convert.py              # (configuration dialog lives in application/tab_analysis/dialogs/unet/)
    │   │   └── unet_bundle/                 # the inference bundle (lazily loads TensorFlow)
    │   │       ├── __main__.py              # CLI: run inference standalone
    │   │       ├── inference.py             # end-to-end predict()
    │   │       ├── model.py                 # Keras U-Net definition
    │   │       ├── preprocessing.py         # input prep
    │   │       ├── field_builder.py         # raw output → VelocityField
    │   │       ├── coordinate_transforms.py # pixel ↔ field-space mapping
    │   │       ├── visualization.py         # diagnostic plots
    │   │       ├── io.py, config.py, cuda_runtime.py
    │   │       └── unet_assets/             # best.weights.h5, meta.csv, norm.json, reference_field.npz
    │   │                                    # (all fetched via scripts/download_assets.py)
    │   │
    │   └── mle/                             # alternative size-distribution methods
    │       ├── ftla_distributor.py          # parametric FTLA (Saveyn 2010)
    │       ├── iterative_distributor.py     # non-parametric iterative (Walker 2012)
    │       ├── likelihood.py                # likelihood functions
    │       ├── families.py                  # parametric family definitions
    │       ├── track_stats.py               # per-track summary stats
    │       └── types.py
    │
    ├── tracking/                            # tracking pipeline (pluggable methods)
    │   ├── __init__.py                      # imports plugin subpackages so their @register_* run
    │   ├── tracking_processor.py            # generic orchestrator; delegates to a TrackingMethod
    │   ├── tracking_method.py               # base class + registry, plus SplitMethod
    │   │                                    # (detector + linker pair)
    │   ├── detection_method.py              # base class + registry for detection algorithms
    │   ├── linking_method.py                # base class + registry for linking algorithms
    │   ├── types.py                         # FolderItem / shared dataclasses
    │   ├── status_codes.py                  # enum codes used by progress emission
    │   │
    │   └── trackmate/                       # TrackMate implementation of TrackingMethod
    │       ├── method.py                    # @register_method, default tracking method
    │       ├── fiji_runner.py               # spawns Fiji and feeds it the Jython script
    │       ├── threshold_calculator.py      # automated thresholding
    │       ├── image_processor.py           # single-TIFF stack normalization
    │       ├── fitting.py                   # quality-histogram model fitting
    │       ├── fitting_models.py            # Cheng-Schwartzman / Poly2 / Gaussian model definitions
    │       ├── types.py                     # FitResult / ThresholdResult dataclasses
    │       └── scripts/                     # Jython scripts executed inside Fiji
    │           ├── tracking.py
    │           └── thresholding.py
    │
    ├── application/                         # Qt controllers / dialogs / output writers
    │   ├── main_window/                     # OpenNtaMainWindow + hand-coded UI builder
    │   │   ├── main_window.py               # OpenNtaMainWindow class
    │   │   ├── branding.py                  # logo / fit-to-view icon (Qt rendering)
    │   │   ├── logo.png                     # app logo
    │   │   ├── _ui_builder.py               # hand-coded Qt widget tree for the main window
    │   │   ├── _ui_constants.py             # shared sizing/font constants
    │   │   ├── _ui_status_bar.py            # status-bar builder
    │   │   └── _ui_tab_tracking.py, _ui_tab_analysis.py, _ui_tab_batch.py  # per-tab builders
    │   ├── config/                          # user settings: ConfigManager store + ConfigDialog
    │   │   ├── dialog.py                    # ConfigDialog
    │   │   ├── manager.py                   # ConfigManager
    │   │   └── _ui_builder.py               # Qt builder mixin for ConfigDialog
    │   ├── common/                          # cross-tab utilities
    │   │   ├── path_selector.py             # native file/folder picker helpers
    │   │   ├── fonts.py                     # cross-platform default Qt font
    │   │   ├── report_templates.py          # HTML report templates + CSS
    │   │   └── statistics.py                # report-side statistics (StatisticsUtils)
    │   ├── tab_analysis/                    # Analysis tab (TabAnalysis)
    │   │   ├── tab.py, ui.py, helper.py, adaptor.py, output.py, _plot_theme.py
    │   │   └── dialogs/                      # corrector configurators (numerical/ + unet/)
    │   ├── tab_batch_analysis/              # Batch tab (TabBatchAnalysis)
    │   │   ├── tab.py, ui.py, helper.py, adaptor.py, output.py, types.py
    │   └── tab_tracking/                    # Track tab (TabTracking)
    │       ├── tab.py, ui.py, helper.py, adaptor.py
    │       └── dialogs/                     # Qt configurator dialogs for tracking methods
    │           ├── __init__.py              # configure_method() dispatcher
    │           └── trackmate/               # TrackMate configurator dialog
    │               ├── dialog.py            # TrackMateDialog
    │               ├── preview_worker.py    # background detection-preview QThread
    │               └── _ui_builder.py, _plot_builder.py, _plot_style.py
    │
    └── common/
        └── progress.py                      # ProgressEmitter — progress-bar plumbing
```

To inspect the actual import graph, `scripts/dependency_drawer.py` walks the
package and produces `opennta_deps.svg`.

For the architecture conventions behind this layout — including which "shared
filename" patterns are intentional and how to plug a new algorithm in via the
existing registries — see [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

### Testing

#### Installing test requirements

Install the development/test extras first:

```sh
pip install -e ".[dev]"
```

#### Test data strategy

The test suite does **not** read pre-recorded sample files from disk.
Every spot CSV, TIFF stack, and Brownian trajectory used by a test is
synthesised in-process from analytic parameters (see
`tests/_helpers/synth.py`) so each test verifies the unit under
test against ground truth it constructed itself, free of imaging or
labelling artefacts. The suite is therefore fully runnable without any
external asset download.

#### Running tests

For most users, **a) Full suite** is sufficient.

> **a) Full suite:** run the entire test suite.
> ```sh
> pytest
> ```
>
> **b) Unit only (skip the slower integration pipeline):**
> ```sh
> pytest -m "not integration"
> ```
>
> **c) Single file:** run all tests in a single file.
> ```sh
> pytest tests/integration/test_analysis_pipeline.py
> ```
>
> **d) Specific test:** run one specific test with verbose output.
> ```sh
> pytest tests/integration/test_analysis_pipeline.py::test_pipeline_recovers_input_diameter -v
> ```

The full suite runs headlessly: PyQt5 and a display server are not required,
because the test conftest stubs the Qt-bearing parent packages and forces a
non-interactive matplotlib backend before any plot helper is imported.

### Logging

Standard `logging` is configured in `main()` with both a console handler and a
rotating file handler. Log files are written under the platform's user log
directory (Windows: `%LOCALAPPDATA%\opennta\logs`; macOS:
`~/Library/Logs/opennta`; Linux: `$XDG_STATE_HOME/opennta/logs` or
`~/.local/state/opennta/logs`). Set `PYTHONLOGLEVEL=DEBUG` (or edit
`__main__.py`) for more verbose output while debugging.

For developer notes — registering a new drift corrector, size distributor, or
tracking method, and the hand-coded Qt UI builder convention — see
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## Roadmap

- Command-line interface
- Direct result transmission to analysis software (e.g., Origin, Prism) via their APIs
- Modularize MSD and R^2 fitting methods
- Count-to-concentration conversion using calibration curves before exporting to analysis software

## Contributing

Contributions, bug reports, and feature requests are welcome via the
[issue tracker](https://github.com/sangjinseo14-prog/opennta/issues). For
architecture conventions, registry-based extension points, and the pull
request workflow, see [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## Releases and tags

Two independent tag namespaces are used:
- `vX.Y.Z` — code releases (semver). The version is derived from
  this git tag by `setuptools-scm` (per the `tag_regex` in
  `pyproject.toml`), written to `src/opennta/_version.py`, and exposed at
  runtime as `opennta.__version__`. When no matching tag is reachable, the
  fallback is `0.0.0+unknown`.
- `assets_vX.Y.Z` — bundled assets (e.g., U-Net inference weights/metadata)
  fetched by `scripts/download_assets.py`. The default tag is set by
  `DEFAULT_TAG` in that script. Assets are versioned independently of the
  code; bumping one does not require bumping the other.

## License

MIT. See [`LICENSE`](LICENSE).

## Contact

Repository: <https://github.com/sangjinseo14-prog/opennta>.
