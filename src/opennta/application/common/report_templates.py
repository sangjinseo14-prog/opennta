from __future__ import annotations

from datetime import datetime


def get_analysis_report_css() -> str:
    return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            font-size: 9pt;
            font-weight: 400;
            line-height: 1.4;
            color: #2c3e50;
            background-color: #ffffff;
            padding: 14px 16px;
            margin: 0;
        }

        h2 {
            color: #1f3a5f;
            font-size: 13pt;
            font-weight: 700;
            margin: 0 0 14px 0;
            padding: 0 0 8px 0;
            border-bottom: 2px solid #3498db;
            letter-spacing: -0.3px;
        }

        h3 {
            color: #ffffff;
            background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
            font-size: 10pt;
            font-weight: 700;
            margin: 0 0 0 0;
            padding: 7px 12px;
            border-radius: 4px 4px 0 0;
            letter-spacing: 0.2px;
        }

        h3 .section-index {
            display: inline-block;
            min-width: 18px;
            height: 18px;
            line-height: 18px;
            text-align: center;
            background: rgba(255,255,255,0.22);
            border-radius: 50%;
            font-size: 9pt;
            font-weight: 700;
            margin-right: 8px;
        }

        h4 {
            color: #34495e;
            font-size: 9pt;
            font-weight: 700;
            margin: 12px 0 5px 0;
            padding-left: 8px;
            border-left: 3px solid #3498db;
            letter-spacing: 0.1px;
        }

        table {
            border-collapse: collapse;
            width: auto;
            table-layout: fixed;
            margin: 0;
            background-color: #ffffff;
        }

        th {
            background-color: #ecf3f9;
            text-align: left;
            padding: 7px 12px;
            border: none;
            border-bottom: 1px solid #d6e2ed;
            font-weight: 400;
            font-size: 9pt;
            color: #5a6c7d;
            letter-spacing: 0.2px;
            width: 160px;
            height: 28px;
            box-sizing: border-box;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        td {
            padding: 7px 12px;
            border: none;
            border-bottom: 1px solid #eef2f5;
            font-size: 9pt;
            font-weight: 400;
            vertical-align: middle;
            width: 160px;
            height: 28px;
            box-sizing: border-box;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        th:first-child,
        td:first-child {
            width: 220px;
        }

        td.label {
            color: #6b7a89;
        }

        tr:last-child td {
            border-bottom: none;
        }

        .section {
            margin-bottom: 16px;
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 4px;
            box-shadow: 0 1px 2px rgba(31, 58, 95, 0.05);
            overflow: hidden;
        }

        .section-body {
            padding: 4px 12px 8px 12px;
        }

        .stat-value {
            font-weight: 400;
            color: #1f3a5f;
            font-size: 9pt;
            font-variant-numeric: tabular-nums;
        }

        .unit {
            color: #8a99a8;
            font-weight: 400;
            font-size: 9pt;
        }

        .small {
            font-size: 8pt;
        }

        .xsmall {
            font-size: 7pt;
        }

        .badge {
            display: inline-block;
            padding: 2px 9px;
            border-radius: 10px;
            background: #eaf3fb;
            color: #1f3a5f;
            font-weight: 400;
            font-size: 9pt;
            letter-spacing: 0.2px;
        }

        .badge.muted {
            background: #f0f2f5;
            color: #6b7a89;
        }

        .peak-table {
            margin: 4px 0 0 0;
        }

        .peak-table th,
        .peak-table td {
            font-size: 9pt;
            padding: 7px 12px;
            height: 28px;
        }

        .peak-table th {
            background-color: #f7fafd;
        }

        .empty {
            padding: 10px 12px;
            color: #8a99a8;
            font-style: italic;
            font-size: 9pt;
        }

        .report-meta {
            margin: -8px 0 14px 0;
            padding: 6px 10px;
            background: #f7fafd;
            border: 1px solid #e1e8ef;
            border-radius: 3px;
            font-size: 8.5pt;
            color: #5a6c7d;
            line-height: 1.5;
        }

        .report-meta .meta-label {
            color: #8a99a8;
            margin-right: 6px;
        }

        .report-meta .meta-value {
            color: #1f3a5f;
            font-variant-numeric: tabular-nums;
            word-break: break-all;
        }

        .report-meta .meta-row {
            display: block;
        }

        .report-meta .meta-file-item {
            padding-left: 14px;
        }
    """


# Sections not listed here are appended in dict order, so new ones surface
# automatically without editing this tuple.
_SECTION_ORDER = ("diameter", "drift", "correction", "msd")


def _ordered_stats_section_keys(stats_dict: dict) -> list[str]:
    keys = list(stats_dict.keys())
    ordered = [k for k in _SECTION_ORDER if k in stats_dict]
    extras = [k for k in keys if k not in _SECTION_ORDER]
    return ordered + extras


# Roughly: data column ~22 chars at 9pt fits in 160px; first column ~30 chars at 220px.
_DATA_CELL_FIT = 20
_DATA_CELL_TIGHT = 26


def _format_number(value, scientific: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    fvalue = float(value)
    if scientific:
        mantissa, exp = f"{fvalue:.1e}".split('e')
        return f"{mantissa}E{int(exp)}"
    return f"{fvalue:.1f}"


def _wrap_with_responsive_size_class(html: str, plain_len: int) -> str:
    if plain_len > _DATA_CELL_TIGHT:
        return f'<span class="xsmall">{html}</span>'
    if plain_len > _DATA_CELL_FIT:
        return f'<span class="small">{html}</span>'
    return html


def _render_value_cell(value, unit: str = "", scientific: bool = False) -> str:
    formatted = _format_number(value, scientific=scientific)
    if unit:
        inner = f'<span class="stat-value">{formatted}</span> <span class="unit">{unit}</span>'
        plain_len = len(formatted) + 1 + len(unit)
    else:
        inner = f'<span class="stat-value">{formatted}</span>'
        plain_len = len(formatted)
    return _wrap_with_responsive_size_class(inner, plain_len)


def _format_distribution_section(key: str, stats: dict, index: int) -> str:
    unit = stats.get('unit', '')
    has_mode = stats.get('mode') is not None
    sample_count = len(stats['data']) if stats.get('data') is not None else 0

    body = f"""
        <h4>Distribution statistics</h4>
        <table>
            <tr>
                <th>Statistic</th>
                <th>Value</th>
            </tr>
            <tr>
                <td class="label">Sample count</td>
                <td>{_render_value_cell(sample_count)}</td>
            </tr>
            <tr>
                <td class="label">Mean</td>
                <td>{_render_value_cell(stats['mean'], unit)}</td>
            </tr>
            <tr>
                <td class="label">Median</td>
                <td>{_render_value_cell(stats['median'], unit)}</td>
            </tr>
            <tr>
                <td class="label">Mode</td>
                <td>{_render_value_cell(stats['mode'], unit) if has_mode else '<span class="stat-value">N/A</span>'}</td>
            </tr>
            <tr>
                <td class="label">Standard deviation</td>
                <td>{_render_value_cell(stats['std'], unit)}</td>
            </tr>
        </table>
    """

    extra = ""
    if key == "diameter" and stats.get("peaks"):
        extra += '<h4>Detected distribution peaks</h4>'
        extra += '<table class="peak-table">'
        extra += (
            '<tr><th>Rank</th><th>Peak mode</th>'
            '<th>Count</th><th>Prominence</th></tr>'
        )
        for i, peak in enumerate(stats["peaks"][:5], 1):
            extra += (
                f'<tr><td>{i}</td>'
                f'<td>{_render_value_cell(peak["mode"], unit)}</td>'
                f'<td>{_render_value_cell(int(peak["count"]))}</td>'
                f'<td>{_render_value_cell(peak["prominence"])}</td></tr>'
            )
        extra += '</table>'

    return f"""
        <div class="section">
            <h3><span class="section-index">{index}</span>{stats['name']}</h3>
            <div class="section-body">
                {body}
                {extra}
            </div>
        </div>
    """


def _format_value_with_unit(value: float, unit: str) -> str:
    if isinstance(value, (int, float)):
        return _render_value_cell(value, unit, scientific=True)
    formatted = str(value)
    unit_html = f' <span class="unit">{unit}</span>' if unit else ""
    return _wrap_with_responsive_size_class(
        f'<span class="stat-value">{formatted}</span>{unit_html}',
        len(formatted) + (1 + len(unit) if unit else 0),
    )


def _format_drift_vector_table(stats: dict) -> str:
    unit = stats.get("unit", "um/s")
    n_tracks = int(stats.get("n_tracks", 0))
    mean_x, mean_y = stats.get("mean_xy", (float("nan"), float("nan")))
    median_x, median_y = stats.get("median_xy", (float("nan"), float("nan")))
    std_x, std_y = stats.get("std_xy", (float("nan"), float("nan")))

    def format_component(v: float) -> str:
        return _format_value_with_unit(float(v), unit)

    return f"""
        <table>
            <tr>
                <th>Statistic</th>
                <th>X component</th>
                <th>Y component</th>
            </tr>
            <tr>
                <td class="label">Tracks (MSD-passed)</td>
                <td colspan="2"><span class="stat-value">{n_tracks}</span></td>
            </tr>
            <tr>
                <td class="label">Mean</td>
                <td>{format_component(mean_x)}</td>
                <td>{format_component(mean_y)}</td>
            </tr>
            <tr>
                <td class="label">Median</td>
                <td>{format_component(median_x)}</td>
                <td>{format_component(median_y)}</td>
            </tr>
            <tr>
                <td class="label">Std dev</td>
                <td>{format_component(std_x)}</td>
                <td>{format_component(std_y)}</td>
            </tr>
        </table>
    """


def _format_drift_section(stats: dict, index: int) -> str:
    return f"""
        <div class="section">
            <h3><span class="section-index">{index}</span>{stats.get('name', 'Drift')}</h3>
            <div class="section-body">
                <h4>Drift vector statistics</h4>
                {_format_drift_vector_table(stats)}
            </div>
        </div>
    """


def _format_correction_section(stats: dict, index: int) -> str:
    label = stats.get("label", "No correction")
    mode = stats.get("mode", "no")
    params = stats.get("params", {}) or {}
    post_drift = stats.get("post_drift", {})

    is_corrected = mode != "no"
    badge_class = "badge" if is_corrected else "badge muted"

    mode_table = f"""
        <h4>Correction mode</h4>
        <table>
            <tr>
                <th>Setting</th>
                <th>Value</th>
            </tr>
            <tr>
                <td class="label">Mode</td>
                <td><span class="{badge_class}">{label}</span></td>
            </tr>
        </table>
    """

    params_html = ""
    if params:
        params_rows = ""
        for pname, pvalue in params.items():
            if isinstance(pvalue, tuple) and len(pvalue) == 2:
                value, unit = pvalue
                if isinstance(value, float):
                    use_sci = abs(value) >= 1e3 or (value != 0 and abs(value) < 1e-2)
                    cell = _render_value_cell(value, unit, scientific=use_sci)
                else:
                    cell = _render_value_cell(value, unit)
                params_rows += (
                    f'<tr>'
                    f'<td class="label">{pname}</td>'
                    f'<td>{cell}</td>'
                    f'</tr>'
                )
            else:
                params_rows += (
                    f'<tr>'
                    f'<td class="label">{pname}</td>'
                    f'<td>{_render_value_cell(pvalue)}</td>'
                    f'</tr>'
                )
        params_html = f"""
            <h4>Corrector parameters</h4>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                {params_rows}
            </table>
        """

    post_drift_html = ""
    if post_drift:
        post_drift_html = (
            '<h4>Post-correction drift</h4>'
            + _format_drift_vector_table(post_drift)
        )

    return f"""
        <div class="section">
            <h3><span class="section-index">{index}</span>{stats['name']}</h3>
            <div class="section-body">
                {mode_table}
                {params_html}
                {post_drift_html}
            </div>
        </div>
    """


def _format_msd_section(stats: dict, index: int) -> str:
    lag_frame = int(stats.get("lag_frame", 0))
    msd_max_lag = int(stats.get("msd_max_lag", 0))
    extra_points = int(stats.get("extra_points", 0))
    min_points = int(stats.get("min_points", 0))
    r2_threshold = float(stats.get("r2_threshold", 0.0))
    passed = int(stats.get("passed", 0))
    failed_d = int(stats.get("failed_d", 0))
    failed_r2 = int(stats.get("failed_r2", 0))
    total = int(stats.get("total", 0))

    body = f"""
        <h4>Fit parameters</h4>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td class="label">Fit lag</td>
                <td>{_render_value_cell(lag_frame, "frames")}</td>
            </tr>
            <tr>
                <td class="label">MSD max lag</td>
                <td>{_render_value_cell(msd_max_lag, "frames")}</td>
            </tr>
            <tr>
                <td class="label">Fit points (lag + extra)</td>
                <td>{_wrap_with_responsive_size_class(
                    f'<span class="stat-value">{lag_frame + extra_points}</span>'
                    f' <span class="unit">(extra = {extra_points}, min = {min_points})</span>',
                    len(str(lag_frame + extra_points)) + len(f' (extra = {extra_points}, min = {min_points})'),
                )}</td>
            </tr>
            <tr>
                <td class="label">R² threshold</td>
                <td>{_render_value_cell(r2_threshold)}</td>
            </tr>
        </table>
        <h4>Track filtering</h4>
        <table>
            <tr>
                <th>Result</th>
                <th>Count</th>
            </tr>
            <tr>
                <td class="label">Tracks passed</td>
                <td>{_wrap_with_responsive_size_class(
                    f'<span class="stat-value">{passed}</span> <span class="unit">/ {total}</span>',
                    len(str(passed)) + len(f' / {total}'),
                )}</td>
            </tr>
            <tr>
                <td class="label">Tracks failed: D&lt;0</td>
                <td>{_wrap_with_responsive_size_class(
                    f'<span class="stat-value">{failed_d}</span> <span class="unit">/ {total}</span>',
                    len(str(failed_d)) + len(f' / {total}'),
                )}</td>
            </tr>
            <tr>
                <td class="label">Tracks failed: R² threshold</td>
                <td>{_wrap_with_responsive_size_class(
                    f'<span class="stat-value">{failed_r2}</span> <span class="unit">/ {total}</span>',
                    len(str(failed_r2)) + len(f' / {total}'),
                )}</td>
            </tr>
        </table>
    """

    return f"""
        <div class="section">
            <h3><span class="section-index">{index}</span>{stats['name']}</h3>
            <div class="section-body">{body}</div>
        </div>
    """


def format_statistics_section(stats_dict: dict) -> str:
    html_parts = []
    for index, key in enumerate(_ordered_stats_section_keys(stats_dict), 1):
        stats = stats_dict[key]
        kind = stats.get("kind")
        if kind == "correction" or key == "correction":
            html_parts.append(_format_correction_section(stats, index))
        elif kind == "drift_vector" or key == "drift":
            html_parts.append(_format_drift_section(stats, index))
        elif kind == "msd" or key == "msd":
            html_parts.append(_format_msd_section(stats, index))
        else:
            html_parts.append(_format_distribution_section(key, stats, index))
    return ''.join(html_parts)


def format_individual_diameter_summary(file_label: str, stats_dict: dict) -> str:
    # Compact per-file block used under the merged group report — only the
    # diameter distribution + peaks, with the file label as the heading and
    # no section-index badge (these aren't part of the main numbered flow).
    diameter = (stats_dict or {}).get("diameter")
    if not diameter:
        return ""

    unit = diameter.get("unit", "")
    has_mode = diameter.get("mode") is not None
    sample_count = len(diameter["data"]) if diameter.get("data") is not None else 0

    stats_table = f"""
        <table>
            <tr>
                <th>Statistic</th>
                <th>Value</th>
            </tr>
            <tr>
                <td class="label">Sample count</td>
                <td>{_render_value_cell(sample_count)}</td>
            </tr>
            <tr>
                <td class="label">Mean</td>
                <td>{_render_value_cell(diameter['mean'], unit)}</td>
            </tr>
            <tr>
                <td class="label">Median</td>
                <td>{_render_value_cell(diameter['median'], unit)}</td>
            </tr>
            <tr>
                <td class="label">Mode</td>
                <td>{_render_value_cell(diameter['mode'], unit) if has_mode else '<span class="stat-value">N/A</span>'}</td>
            </tr>
            <tr>
                <td class="label">Standard deviation</td>
                <td>{_render_value_cell(diameter['std'], unit)}</td>
            </tr>
        </table>
    """

    peaks_html = ""
    if diameter.get("peaks"):
        peaks_rows = ""
        for i, peak in enumerate(diameter["peaks"][:5], 1):
            peaks_rows += (
                f'<tr><td>{i}</td>'
                f'<td>{_render_value_cell(peak["mode"], unit)}</td>'
                f'<td>{_render_value_cell(int(peak["count"]))}</td>'
                f'<td>{_render_value_cell(peak["prominence"])}</td></tr>'
            )
        peaks_html = f"""
            <h4>Detected distribution peaks</h4>
            <table class="peak-table">
                <tr>
                    <th>Rank</th>
                    <th>Peak mode</th>
                    <th>Count</th>
                    <th>Prominence</th>
                </tr>
                {peaks_rows}
            </table>
        """

    return f"""
        <div class="section">
            <h3>{file_label}</h3>
            <div class="section-body">
                <h4>Distribution statistics</h4>
                {stats_table}
                {peaks_html}
            </div>
        </div>
    """


def _format_report_meta_rows(
    file_path: str | list[str] | None, created_at: datetime | None
) -> str:
    rows = []

    if created_at is None:
        created_at = datetime.now()
    date_part = created_at.strftime("%m/%d/%Y")
    hour = str(int(created_at.strftime("%I")))
    time_part = f"{hour}:{created_at.strftime('%M %p')}"
    timestamp = f"{date_part} {time_part}"

    rows.append(
        f'<span class="meta-row">'
        f'<span class="meta-label">Created:</span>'
        f'<span class="meta-value">{timestamp}</span>'
        f'</span>'
    )

    if isinstance(file_path, str):
        paths = [file_path]
    elif file_path:
        paths = [p for p in file_path if p]
    else:
        paths = []

    if len(paths) == 1:
        rows.append(
            f'<span class="meta-row">'
            f'<span class="meta-label">File:</span>'
            f'<span class="meta-value">{paths[0]}</span>'
            f'</span>'
        )
    elif len(paths) > 1:
        rows.append(
            '<span class="meta-row">'
            '<span class="meta-label">Files:</span>'
            '</span>'
        )
        for p in paths:
            rows.append(
                f'<span class="meta-row meta-file-item">'
                f'<span class="meta-value">{p}</span>'
                f'</span>'
            )

    return f'<div class="report-meta">{"".join(rows)}</div>'


def get_analysis_report_template(
    stats_dict: dict,
    title: str = "Statistical analysis report",
    file_path: str | list[str] | None = None,
    created_at: datetime | None = None,
    extra_html: str = "",
) -> str:
    if not stats_dict:
        return "<html><body><p>No statistics available.</p></body></html>"

    css = get_analysis_report_css()
    statistics_html = format_statistics_section(stats_dict)
    meta_html = _format_report_meta_rows(file_path, created_at)

    html = f"""<html>
<head>
    <style>
{css}
    </style>
</head>
<body>
<h2>{title}</h2>
{meta_html}
{statistics_html}
{extra_html}
</body>
</html>"""

    return html
