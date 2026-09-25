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
from pathlib import Path
import mne
import subprocess, tempfile, webbrowser

_fsaverage_coords = None

def fsaverage_coordinates():
    global _fsaverage_coords
    if _fsaverage_coords is None:
        fs = mne.datasets.fetch_fsaverage(verbose=False)
        src = mne.read_source_spaces(str(fs) + '/bem/fsaverage-ico-5-src.fif', verbose=False)
        _fsaverage_coords = np.vstack([src[0]['rr'][src[0]['vertno']], src[1]['rr'][src[1]['vertno']]])
    return _fsaverage_coords


def _resolve_path(path):
    if ":" in path and "@" in path.split(":")[0]:
        suffix = Path(path.split(":")[-1]).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        print(f"Fetching {path}...")
        subprocess.run(["scp", path, tmp.name], check=True)
        print(f"  -> {tmp.name}")
        return tmp.name
    return path


def load(path):
    df = pd.read_csv(_resolve_path(path))
    if "electrode" in df.columns:
        return [df], "electrode"
    nums = df.select_dtypes("number").columns.tolist()
    value_cols = [c for c in nums if c.startswith("value")]
    if len(value_cols) > 1:
        coord_cols = [c for c in nums if c not in value_cols]
        frames = []
        for vc in sorted(value_cols):
            sub = df[coord_cols + [vc]].copy()
            sub.columns = ["x", "y", "z", "value"]
            frames.append(sub)
        return frames, "xyz"
    if "value" in df.columns:
        val_col = "value"
        coord_cols = [c for c in nums if c != "value"]
    else:
        coord_cols = nums[:3]
        val_col = nums[3]
    sub = df[coord_cols + [val_col]].copy()
    sub.columns = ["x", "y", "z", "value"]
    return [sub], "xyz"

def get_montage_positions():
    montage = mne.channels.make_standard_montage("standard_1005")
    positions = montage.get_positions()["ch_pos"]
    pos_2d = {}
    for ch, xyz in positions.items():
        pos_2d[ch.upper()] = (xyz[0], xyz[1])
    return pos_2d


# linear ramp across the top quartile, replacing the old discrete tiers
GRADIENT_QUANTILE = 0.75
GRADIENT_COLORSCALE = [
    [0.00, "#a3e635"],
    [0.33, "#eab308"],
    [0.67, "#ea580c"],
    [1.00, "#dc2626"],
]


def _gradient_sizes(vals, lo, hi, smin, smax):
    """Linear marker sizes over [lo, hi]; flat at smax if the range degenerates."""
    if hi <= lo:
        return np.full(len(vals), smax)
    t = np.clip((np.asarray(vals, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    return smin + t * (smax - smin)


def make_3d_traces(df, name):
    lo = df.value.quantile(GRADIENT_QUANTILE)
    hi = df.value.max()

    rest = df[df.value < lo]
    top = df[df.value >= lo]

    hover = "x:%{x:.1f} y:%{y:.1f} z:%{z:.1f}<br>value:%{customdata:.3f}<extra></extra>"

    traces = [go.Scatter3d(
        x=rest.x, y=rest.y, z=rest.z,
        mode="markers",
        marker=dict(size=2, color="#9ca3af", opacity=0.15),
        customdata=rest.value,
        name=f"{name} <75% (< {lo:.2f})",
        hovertemplate=hover,
    )]

    if not top.empty:
        traces.append(go.Scatter3d(
            x=top.x, y=top.y, z=top.z,
            mode="markers",
            marker=dict(
                size=_gradient_sizes(top.value, lo, hi, 2.5, 5),
                color=top.value,
                colorscale=GRADIENT_COLORSCALE,
                cmin=lo, cmax=hi,
                opacity=0.9,
                colorbar=dict(title=f"{name}<br>top 25%", thickness=12, len=0.6),
            ),
            customdata=top.value,
            name=f"{name} top 25% ({lo:.2f}-{hi:.2f})",
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
    lo = np.quantile(vals, GRADIENT_QUANTILE)
    hi = vals.max()

    xs = np.array(xs)
    ys = np.array(ys)

    hover = "%{text}<br>value:%{customdata:.3f}<extra></extra>"
    traces = []
    for mask, size, color, opacity, label in [
        (vals < p50,                   6, "#d1d5db", 0.3, f"{name} <50% (< {p50:.2f})"),
        ((vals >= p50) & (vals < lo),  8, "#9ca3af", 0.5, f"{name} top 25-50% ({p50:.2f}-{lo:.2f})"),
    ]:
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

    top = vals >= lo
    if top.any():
        traces.append(go.Scatter(
            x=xs[top], y=ys[top],
            mode="markers+text",
            marker=dict(
                size=_gradient_sizes(vals[top], lo, hi, 10, 20),
                color=vals[top],
                colorscale=GRADIENT_COLORSCALE,
                cmin=lo, cmax=hi,
                colorbar=dict(title=f"{name}<br>top 25%", thickness=12, len=0.6),
            ),
            text=labels[top],
            textposition="top center",
            textfont=dict(size=7),
            customdata=vals[top],
            name=f"{name} top 25% ({lo:.2f}-{hi:.2f})",
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


def _make_colony_fig(colony, coordinates, name, sign):
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
    return fig



def show_colony(colony, coordinates=None, name="colony", with_pos=True, with_neg=False, output=None):
    from colony import MultiColony
    if coordinates is None:
        coordinates = fsaverage_coordinates()
    signs = []
    if with_pos:
        signs.append("pos")
    if with_neg:
        signs.append("neg")
    if isinstance(colony, MultiColony):
        return show_multicolony(colony, coordinates, name, signs, output)
    figs = []
    for sign in signs:
        figs.append((f"{name}_{sign}", _make_colony_fig(colony, coordinates, name, sign)))
    if output:
        _write_tabbed_html(figs, output)
    else:
        for _, fig in figs:
            fig.show()
    return figs


def show_multicolony(multi, coordinates=None, name="colony", signs=("pos", "neg"), output=None):
    if coordinates is None:
        coordinates = fsaverage_coordinates()
    figs = []
    for sign in signs:
        if not getattr(multi.colonies[0], f"include_{sign}", False):
            continue
        for i, colony in enumerate(multi.colonies):
            t_ms = int(i * multi.interval * 1000)
            label = f"{name}_{sign}_{t_ms}ms"
            figs.append((label, _make_colony_fig(colony, coordinates, f"{name} t={t_ms}ms", sign)))

    if not output:
        output = f"colony_{name}.html"
    _write_tabbed_html(figs, output)
    return figs


def show_colonies(colonies: list[tuple[str, "Colony | MultiColony"]], coordinates=None, signs=("pos", "neg"), output=None):
    from colony import MultiColony
    if coordinates is None:
        coordinates = fsaverage_coordinates()
    figs = []
    for name, colony in colonies:
        if isinstance(colony, MultiColony):
            for sign in signs:
                if not getattr(colony.colonies[0], f"include_{sign}", False):
                    continue
                for i, sub in enumerate(colony.colonies):
                    t_ms = int(i * colony.interval * 1000)
                    label = f"{name}_{sign}_{t_ms}ms"
                    figs.append((label, _make_colony_fig(sub, coordinates, f"{name} t={t_ms}ms", sign)))
        else:
            for sign in signs:
                if not getattr(colony, f"include_{sign}", False):
                    continue
                figs.append((f"{name}_{sign}", _make_colony_fig(colony, coordinates, name, sign)))

    if output:
        _write_tabbed_html(figs, output)
    else:
        for _, fig in figs:
            fig.show()
    return figs


def _write_tabbed_html(figs, output_path):
    FIG_HEIGHT = 700
    tabs_html = []
    divs_html = []
    for i, (label, fig) in enumerate(figs):
        fig.update_layout(height=FIG_HEIGHT, autosize=True)
        div_id = f"tab-{i}"
        active = "active" if i == 0 else ""
        display = "block" if i == 0 else "none"
        tabs_html.append(f'<button class="tab-btn {active}" onclick="switchTab({i})">{label}</button>')
        divs_html.append(f'<div id="{div_id}" class="tab-content" style="display:{display}">{fig.to_html(full_html=False, include_plotlyjs=(i == 0))}</div>')

    n = len(figs)
    html = f"""<!DOCTYPE html><html><head><style>
    body {{ margin: 0; }}
    .tab-bar {{ display: flex; flex-wrap: wrap; gap: 2px; align-items: center; padding: 4px; }}
    .tab-btn {{ padding: 8px 16px; cursor: pointer; border: 1px solid #ccc; background: #f0f0f0; }}
    .tab-btn.active {{ background: #fff; border-bottom: 2px solid #333; }}
    .sync-btn {{ padding: 8px 16px; cursor: pointer; border: 1px solid #69b; background: #def; margin-left: auto; }}
    .tab-content {{ height: {FIG_HEIGHT}px; }}
    </style></head><body>
    <div class="tab-bar">{"".join(tabs_html)}<button class="sync-btn" onclick="syncCamera()">Sync Camera</button></div>
    {"".join(divs_html)}
    <script>
    var savedCamera = null;
    function getPlotDiv(idx) {{
        var tab = document.getElementById('tab-' + idx);
        return tab ? tab.querySelector('.plotly-graph-div') : null;
    }}
    function switchTab(idx) {{
        document.querySelectorAll('.tab-content').forEach(function(d, i) {{ d.style.display = i === idx ? 'block' : 'none'; }});
        document.querySelectorAll('.tab-btn').forEach(function(b, i) {{ b.className = 'tab-btn' + (i === idx ? ' active' : ''); }});
        var div = getPlotDiv(idx);
        if (div) {{
            requestAnimationFrame(function() {{
                Plotly.Plots.resize(div);
                if (savedCamera) Plotly.relayout(div, {{'scene.camera': savedCamera}});
            }});
        }}
    }}
    function syncCamera() {{
        var active = -1;
        document.querySelectorAll('.tab-content').forEach(function(d, i) {{ if (d.style.display !== 'none') active = i; }});
        if (active < 0) return;
        var div = getPlotDiv(active);
        if (!div || !div.layout || !div.layout.scene) return;
        savedCamera = JSON.parse(JSON.stringify(div.layout.scene.camera));
        var others = [];
        for (var i = 0; i < {n}; i++) {{ if (i !== active) others.push(i); }}
        (function applyNext(list, cb) {{
            if (!list.length) {{ cb(); return; }}
            var idx = list.shift();
            var tab = document.getElementById('tab-' + idx);
            tab.style.display = 'block';
            requestAnimationFrame(function() {{
                var d = getPlotDiv(idx);
                if (d) {{
                    Plotly.Plots.resize(d);
                    Plotly.relayout(d, {{'scene.camera': savedCamera}}).then(function() {{
                        tab.style.display = 'none';
                        applyNext(list, cb);
                    }});
                }} else {{
                    tab.style.display = 'none';
                    applyNext(list, cb);
                }}
            }});
        }})(others, function() {{ alert('Camera synced to all tabs'); }});
    }}
    </script></body></html>"""

    out = Path(output_path).resolve()
    out.write_text(html)
    print(f"saved {out}")
    webbrowser.open(f"file://{out}")


def _build_single_fig(mode, df, name, montage_pos=None):
    fig = go.Figure()
    density_lines = []

    if mode == "inverse":
        all_coords = df[["x", "y", "z"]].values
        for trace in make_3d_traces(df, name):
            fig.add_trace(trace)
        for pct_label, threshold in [("top 5%", 0.95), ("top 10%", 0.90), ("top 15%", 0.85), ("top 25%", 0.75)]:
            subset = df[df.value >= df.value.quantile(threshold)]
            d = vertex_density(subset[["x", "y", "z"]].values, all_coords)
            density_lines.append(f"{name} {pct_label}: {d:.3f}")
        fig.update_layout(
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
            title=f"Source-space colony — {name}",
            margin=dict(l=0, r=0, t=80, b=0),
            legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
        )
    else:
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
            title=f"Electrode colony ({mode}) — {name}",
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


def _build_figs_from_paths(mode, paths):
    montage_pos = get_montage_positions() if mode != "inverse" else None
    figs = []
    for path in paths:
        frames, _ = load(path)
        base_name = path.rsplit("/", 1)[-1].replace(".csv", "")
        if len(frames) == 1:
            figs.append((base_name, _build_single_fig(mode, frames[0], base_name, montage_pos)))
        else:
            for i, df in enumerate(frames):
                label = f"{base_name}_t{i}"
                figs.append((label, _build_single_fig(mode, df, label, montage_pos)))
    return figs


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <inverse|vol|csd> <file1.csv> [file2.csv ...]")
        print(f"       {sys.argv[0]} <inverse|vol|csd> --tabs <file1.csv> <file2.csv> ...")
        sys.exit(1)

    mode = sys.argv[1]
    paths = sys.argv[2:]

    figs = _build_figs_from_paths(mode, paths)

    output_name = f"colony_viewer_{mode}.html"
    if len(figs) == 1:
        figs[0][1].write_html(output_name, include_plotlyjs=True)
        figs[0][1].show()
    else:
        _write_tabbed_html(figs, output_name)

    print(f"saved {output_name}")
