"""
Viewer for colony gain data with percentile highlighting.
- inverse (x,y,z,value): 3D scatter plot
- vol/csd (electrode,value): 2D topomap using 10-05 montage

Usage:
    python colony_viewer.py inverse alpha.csv beta.csv
    python colony_viewer.py vol alpha.csv
    python colony_viewer.py csd alpha.csv beta.csv

Thanks Claude.
"""
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import KDTree
import mne

_fsaverage_coords = None

def fsaverage_coordinates():
    global _fsaverage_coords
    if _fsaverage_coords is None:
        fs = mne.datasets.fetch_fsaverage(verbose=False)
        src = mne.read_source_spaces(str(fs) + '/bem/fsaverage-ico-5-src.fif', verbose=False)
        _fsaverage_coords = np.vstack([src[0]['rr'][src[0]['vertno']], src[1]['rr'][src[1]['vertno']]])
    return _fsaverage_coords


def load(path):
    df = pd.read_csv(path)
    if "electrode" in df.columns:
        return df, "electrode"
    nums = df.select_dtypes("number").columns.tolist()
    if "value" in df.columns:
        val_col = "value"
        coord_cols = [c for c in nums if c != "value"]
    else:
        coord_cols = nums[:3]
        val_col = nums[3]
    df = df[coord_cols + [val_col]].copy()
    df.columns = ["x", "y", "z", "value"]
    return df, "xyz"

def get_montage_positions():
    montage = mne.channels.make_standard_montage("standard_1005")
    positions = montage.get_positions()["ch_pos"]
    pos_2d = {}
    for ch, xyz in positions.items():
        pos_2d[ch.upper()] = (xyz[0], xyz[1])
    return pos_2d


def make_3d_traces(df, name):
    p75 = df.value.quantile(0.75)
    p85 = df.value.quantile(0.85)
    p90 = df.value.quantile(0.90)
    p95 = df.value.quantile(0.95)

    rest   = df[df.value < p75]
    top_25 = df[(df.value >= p75) & (df.value < p85)]
    top_15 = df[(df.value >= p85) & (df.value < p90)]
    top_10 = df[(df.value >= p90) & (df.value < p95)]
    top_5  = df[df.value >= p95]

    hover = "x:%{x:.1f} y:%{y:.1f} z:%{z:.1f}<br>value:%{customdata:.3f}<extra></extra>"

    traces = []
    for subset, size, color, opacity, label in [
        (rest,   2, "#9ca3af", 0.15, f"{name} <75% (< {p75:.2f})"),
        (top_25, 2.5, "#a3e635", 0.40, f"{name} top 15-25% ({p75:.2f}-{p85:.2f})"),
        (top_15, 3, "#eab308", 0.70, f"{name} top 10-15% ({p85:.2f}-{p90:.2f})"),
        (top_10, 4, "#ea580c", 0.85, f"{name} top 5-10% ({p90:.2f}-{p95:.2f})"),
        (top_5,  5, "#dc2626", 0.95, f"{name} top 5% (> {p95:.2f})"),
    ]:
        traces.append(go.Scatter3d(
            x=subset.x, y=subset.y, z=subset.z,
            mode="markers",
            marker=dict(size=size, color=color, opacity=opacity),
            customdata=subset.value,
            name=label,
            hovertemplate=hover,
        ))
    return traces


def make_2d_traces(df, name, montage_pos):
    df = df.copy()
    df["electrode"] = df["electrode"].str.upper()
    matched = df[df["electrode"].isin(montage_pos)]
    if matched.empty:
        print(f"Warning: no electrodes matched 10-05 montage for {name}")
        return []

    xs = [montage_pos[e][0] for e in matched["electrode"]]
    ys = [montage_pos[e][1] for e in matched["electrode"]]
    vals = matched["value"].values
    labels = matched["electrode"].values

    p50 = np.quantile(vals, 0.50)
    p75 = np.quantile(vals, 0.75)
    p85 = np.quantile(vals, 0.85)
    p90 = np.quantile(vals, 0.90)
    p95 = np.quantile(vals, 0.95)

    xs = np.array(xs)
    ys = np.array(ys)

    tiers = [
        (vals < p50,                    6,  "#d1d5db", 0.3,  f"{name} <50% (< {p50:.2f})"),
        ((vals >= p50) & (vals < p75),  8,  "#9ca3af", 0.5,  f"{name} top 25-50% ({p50:.2f}-{p75:.2f})"),
        ((vals >= p75) & (vals < p85), 10, "#a3e635", 0.7,  f"{name} top 15-25% ({p75:.2f}-{p85:.2f})"),
        ((vals >= p85) & (vals < p90), 13, "#eab308", 0.8,  f"{name} top 10-15% ({p85:.2f}-{p90:.2f})"),
        ((vals >= p90) & (vals < p95), 16, "#ea580c", 0.9,  f"{name} top 5-10% ({p90:.2f}-{p95:.2f})"),
        (vals >= p95,                  20, "#dc2626", 1.0,  f"{name} top 5% (> {p95:.2f})"),
    ]

    hover = "%{text}<br>value:%{customdata:.3f}<extra></extra>"
    traces = []
    for mask, size, color, opacity, label in tiers:
        if not mask.any():
            continue
        traces.append(go.Scatter(
            x=xs[mask], y=ys[mask],
            mode="markers+text",
            marker=dict(size=size, color=color, opacity=opacity),
            text=labels[mask],
            textposition="top center",
            textfont=dict(size=7),
            customdata=vals[mask],
            name=label,
            hovertemplate=hover,
        ))
    return traces


def vertex_density(subset_coords, all_coords):
    if len(subset_coords) < 2:
        return 0.0
    all_tree = KDTree(all_coords)
    mesh_edge = all_tree.query(all_coords, k=2)[0][:, 1].mean()
    equidistant_nn = mesh_edge * np.sqrt(len(all_coords) / len(subset_coords))
    sub_tree = KDTree(subset_coords)
    mean_nn = sub_tree.query(subset_coords, k=2)[0][:, 1].mean()
    return float(np.clip(1 - mean_nn / equidistant_nn, 0, 1))


def add_head_outline(fig):
    theta = np.linspace(0, 2 * np.pi, 100)
    r = 0.095
    fig.add_trace(go.Scatter(
        x=r * np.cos(theta), y=r * np.sin(theta),
        mode="lines", line=dict(color="gray", width=1),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[r + 0.008],
        mode="markers", marker=dict(size=6, symbol="triangle-up", color="gray"),
        showlegend=False, hoverinfo="skip",
    ))


def _colony_to_df(colony, coordinates, sign):
    values = getattr(colony, f"colony_{sign}")
    df = pd.DataFrame({"x": coordinates[:, 0], "y": coordinates[:, 1], "z": coordinates[:, 2], "value": values})
    return df


def show_colony(colony, coordinates=None, name="colony", signs=("pos", "neg"), output=None):
    if coordinates is None:
        coordinates = fsaverage_coordinates()
    from plotly.subplots import make_subplots
    figs = []
    for sign in signs:
        if not getattr(colony, f"include_{sign}", False):
            continue
        df = _colony_to_df(colony, coordinates, sign)
        fig = go.Figure()
        all_coords = df[["x", "y", "z"]].values
        for trace in make_3d_traces(df, f"{name}_{sign}"):
            fig.add_trace(trace)
        density_lines = []
        for pct_label, threshold in [("top 5%", 0.95), ("top 10%", 0.90), ("top 15%", 0.85), ("top 25%", 0.75)]:
            subset = df[df.value >= df.value.quantile(threshold)]
            d = vertex_density(subset[["x", "y", "z"]].values, all_coords)
            density_lines.append(f"{pct_label}: {d:.3f}")
        fig.update_layout(
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
            title=f"{name} ({sign})",
            margin=dict(l=0, r=0, t=80, b=0),
            legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
        )
        if density_lines:
            fig.add_annotation(text="Density: " + " | ".join(density_lines),
                               xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False, font=dict(size=11))
        figs.append((f"{name}_{sign}", fig))

    if output:
        _write_tabbed_html(figs, output)
    else:
        for _, fig in figs:
            fig.show()
    return figs


def show_colonies(colonies: list[tuple[str, "Colony"]], coordinates=None, signs=("pos", "neg"), output=None):
    if coordinates is None:
        coordinates = fsaverage_coordinates()
    figs = []
    for name, colony in colonies:
        for sign in signs:
            if not getattr(colony, f"include_{sign}", False):
                continue
            df = _colony_to_df(colony, coordinates, sign)
            fig = go.Figure()
            all_coords = df[["x", "y", "z"]].values
            for trace in make_3d_traces(df, f"{name}_{sign}"):
                fig.add_trace(trace)
            density_lines = []
            for pct_label, threshold in [("top 5%", 0.95), ("top 10%", 0.90), ("top 15%", 0.85), ("top 25%", 0.75)]:
                subset = df[df.value >= df.value.quantile(threshold)]
                d = vertex_density(subset[["x", "y", "z"]].values, all_coords)
                density_lines.append(f"{pct_label}: {d:.3f}")
            fig.update_layout(
                scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
                title=f"{name} ({sign})",
                margin=dict(l=0, r=0, t=80, b=0),
                legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
            )
            if density_lines:
                fig.add_annotation(text="Density: " + " | ".join(density_lines),
                                   xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False, font=dict(size=11))
            figs.append((f"{name}_{sign}", fig))

    if output:
        _write_tabbed_html(figs, output)
    else:
        for _, fig in figs:
            fig.show()
    return figs


def _write_tabbed_html(figs, output_path):
    tabs_html = []
    divs_html = []
    for i, (label, fig) in enumerate(figs):
        div_id = f"tab-{i}"
        active = "active" if i == 0 else ""
        display = "block" if i == 0 else "none"
        tabs_html.append(f'<button class="tab-btn {active}" onclick="switchTab({i})">{label}</button>')
        divs_html.append(f'<div id="{div_id}" class="tab-content" style="display:{display}">{fig.to_html(full_html=False, include_plotlyjs=(i == 0))}</div>')

    html = f"""<!DOCTYPE html><html><head><style>
    .tab-btn {{ padding: 8px 16px; cursor: pointer; border: 1px solid #ccc; background: #f0f0f0; }}
    .tab-btn.active {{ background: #fff; border-bottom: 2px solid #333; }}
    .tab-content {{ width: 100%; }}
    </style></head><body>
    <div>{"".join(tabs_html)}</div>
    {"".join(divs_html)}
    <script>
    function switchTab(idx) {{
        document.querySelectorAll('.tab-content').forEach((d, i) => d.style.display = i === idx ? 'block' : 'none');
        document.querySelectorAll('.tab-btn').forEach((b, i) => b.className = 'tab-btn' + (i === idx ? ' active' : ''));
    }}
    </script></body></html>"""

    Path(output_path).write_text(html)
    print(f"saved {output_path}")


def _build_fig_from_paths(mode, paths):
    fig = go.Figure()
    density_lines = []

    if mode == "inverse":
        for path in paths:
            df, _ = load(path)
            name = path.rsplit("/", 1)[-1].replace(".csv", "")
            all_coords = df[["x", "y", "z"]].values
            for trace in make_3d_traces(df, name):
                fig.add_trace(trace)
            for pct_label, threshold in [("top 5%", 0.95), ("top 10%", 0.90), ("top 15%", 0.85), ("top 25%", 0.75)]:
                subset = df[df.value >= df.value.quantile(threshold)]
                d = vertex_density(subset[["x", "y", "z"]].values, all_coords)
                density_lines.append(f"{name} {pct_label}: {d:.3f}")
        fig.update_layout(
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
            title="Source-space colony — " + ", ".join(paths),
            margin=dict(l=0, r=0, t=80, b=0),
            legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
        )
    else:
        montage_pos = get_montage_positions()
        for path in paths:
            df, _ = load(path)
            name = path.rsplit("/", 1)[-1].replace(".csv", "")
            for trace in make_2d_traces(df, name, montage_pos):
                fig.add_trace(trace)
            matched = df.copy()
            matched["electrode"] = matched["electrode"].str.upper()
            matched = matched[matched["electrode"].isin(montage_pos)]
            if not matched.empty:
                all_coords = np.array([montage_pos[e] for e in matched["electrode"]])
                for pct_label, threshold in [("top 5%", 0.95), ("top 10%", 0.90), ("top 15%", 0.85), ("top 25%", 0.75)]:
                    subset_mask = matched["value"] >= matched["value"].quantile(threshold)
                    subset_coords = np.array([montage_pos[e] for e in matched.loc[subset_mask, "electrode"]])
                    d = vertex_density(subset_coords, all_coords)
                    density_lines.append(f"{name} {pct_label}: {d:.3f}")
        add_head_outline(fig)
        fig.update_layout(
            title=f"Electrode colony ({mode}) — " + ", ".join(paths),
            xaxis=dict(scaleanchor="y", showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=80, b=0),
            legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
            plot_bgcolor="white",
        )

    if density_lines:
        fig.add_annotation(text="Density: " + " | ".join(density_lines),
                           xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False, font=dict(size=11))
    return fig


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <inverse|vol|csd> <file1.csv> [file2.csv ...]")
        print(f"       {sys.argv[0]} <inverse|vol|csd> --tabs <file1.csv> <file2.csv> ...")
        sys.exit(1)

    mode = sys.argv[1]
    args = sys.argv[2:]

    if "--tabs" in args:
        args.remove("--tabs")
        figs = []
        for path in args:
            name = path.rsplit("/", 1)[-1].replace(".csv", "")
            fig = _build_fig_from_paths(mode, [path])
            figs.append((name, fig))
        _write_tabbed_html(figs, f"colony_viewer_{mode}.html")
    else:
        fig = _build_fig_from_paths(mode, args)
        output_name = f"colony_viewer_{mode}.html"
        fig.write_html(output_name, include_plotlyjs=True)
        fig.show()
        print(f"saved {output_name}")
