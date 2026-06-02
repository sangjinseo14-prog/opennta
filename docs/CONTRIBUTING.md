# Contributing to OpenNTA

Thanks for contributing to OpenNTA.
This guide explains where code should go, how we expect it to look,
and how to submit changes cleanly.

For setup, project layout, and tests, start with the [README](../README.md).
For bugs or feature requests, use the
[issue tracker](https://github.com/sangjinseo14-prog/opennta/issues).

## Architecture conventions

Some choices in this repo can look repetitive at first.
They are mostly intentional, so please read this section before “cleaning up” structure.

### Extension is registry-first, not inheritance-heavy

Shared extension points (`DriftCorrector`, `SizeDistributor`, `TrackingMethod`,
`DetectionMethod`, `LinkingMethod`) use a small registry pattern.
You add a strategy with a decorator; the host discovers it automatically.

```text
analysis/
├── drift_corrector.py           # base class + @register_corrector + get_drift_corrector()
├── numerical_field/
│   └── numerical_corrector.py   # @register_corrector — appears in Analysis tab combo
└── unet_field/
    └── unet_corrector.py        # @register_corrector — same, no host-side change
```

In practice: drop in a new file, register the class, and avoid host-side edits.

### Keep logic local to each algorithm module

Algorithm modules such as `tracking/trackmate/`, `analysis/numerical_field/`,
and `analysis/mle/` keep their own `types.py` alongside the
package-level one at `analysis/types.py` or
`tracking/types.py`.

Same filename does **not** mean shared semantics.
These are domain-local definitions and should usually stay that way.

```text
analysis/
├── types.py              # cross-analysis defaults
├── status_codes.py                  # shared analysis status codes
├── numerical_field/
│   └── types.py          # numerical-field-specific params
└── mle/
    └── types.py          # MLE-specific: likelihood / family params

tracking/
├── types.py              # cross-tracking defaults
├── status_codes.py                  # shared tracking status codes
└── trackmate/
    └── types.py          # TrackMate-specific: spot/quality params
```

### Underscore-prefixed files are private by design

Files such as `_ui_builder.py`, `_plot_style.py`, and `_kernel_preview.py`
appear in several dialog folders under `application/tab_*/dialogs/`,
but they are meant to diverge.

Treat them as module-private implementation details.
If you add a new algorithm module, copying one as a starting point is fine;
importing these across module boundaries is usually not.

### Tabs are independent

Top-level tabs are not built around a common `BaseTab` abstraction.
That is intentional: each tab can evolve around its own workflow.

Inside a tab, file splits are by role:

```text
application/tab_analysis/   # also tab_tracking/, tab_batch_analysis/
├── tab.py       # workflow orchestration
├── ui.py        # widgets + signal/slot wiring
├── helper.py    # file pickers, path helpers, small utilities
├── adaptor.py   # bridge to analysis/ (or tracking/) on a worker thread
└── output.py    # analysis/batch tabs only: writes results (reports, diameter/field/corrected CSVs)
```

`tab_analysis/` and `tab_batch_analysis/` each own their `adaptor.py`
(`tab_batch_analysis/adaptor.py` runs the multi-file `BatchAnalysisWorker`).
Logic that is genuinely Qt-free and shared by both — corrected-track export,
R²/quality filtering — lives in `analysis/` (`corrected_track_exporter.py`,
`result_filters.py`) rather than being reached across tab boundaries.

`application/` should orchestrate UI-driven workflows.
Heavy computation belongs in `analysis/` and `tracking/`.

### If unsure, prefer copying over premature abstraction

Two similar local files are often cheaper than a forced shared base.
If a stable shared shape appears later (usually after 3+ concrete cases),
extract it then.

## Code style

These two guides apply to all modules:

- [`COMMENT_STYLE.md`](COMMENT_STYLE.md): keep comments minimal and meaningful.
- [`NAMING_STYLE.md`](NAMING_STYLE.md): names should reveal intent and outcomes.

Please skim both before adding new identifiers or comments.

## Adding a new algorithm

### Drift corrector or size distributor

Subclass `DriftCorrector` (or `SizeDistributor`), set `mode_key` and `ui_label`,
then decorate with `@register_corrector` (or `@register_distributor`).
The corresponding UI combo will discover it automatically.

If your strategy needs a custom dialog/preview, keep those files local in its module folder
(similar to `analysis/numerical_field/` or `analysis/unet_field/`).

### Tracking method

Tracking has three extension points in `src/opennta/tracking/`:

- End-to-end pipeline: subclass `TrackingMethod`, decorate with `@register_method`.
- Detector-only: subclass `DetectionMethod`, decorate with `@register_detector`.
- Linker-only: subclass `LinkingMethod`, decorate with `@register_linker`.

`SplitMethod` can combine any registered detector with any registered linker.

## Editing the Qt UI

The UI is built in code, not loaded from `.ui` files. Each window/dialog
ships an `_ui_builder.py` mixin that constructs its widget tree with
pixel-sized fonts (so the layout looks the same on macOS and Windows
without runtime point→pixel conversion).

- Main window: `src/opennta/application/main_window/_ui_builder.py` plus
  the per-section files `_ui_status_bar.py`, `_ui_tab_tracking.py`,
  `_ui_tab_analysis.py`, `_ui_tab_batch.py`.
- Dialogs follow the same convention — see
  `src/opennta/application/config/_ui_builder.py` for a small
  canonical example.

Widget attribute names are part of the public surface (tab modules
reach into the main window via `self.<name>` access), so always
allocate widgets through the shared `self._register("<name>", widget)`
helper, which sets both the Python attribute and `objectName` in one
step.

## Submitting a pull request

1. Fork and create a branch from `main`.
2. Implement your changes.
3. Run tests (README has the testing section).
4. Open a PR explaining what changed and why.

If you touched a registry or added a new module, mention the entry point
in the PR description so reviewers can find it quickly.
