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
import mne


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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <inverse|vol|csd> <file1.csv> [file2.csv ...]")
        sys.exit(1)

    mode = sys.argv[1]
    paths = sys.argv[2:]

    fig = go.Figure()

    if mode == "inverse":
        for path in paths:
            df, _ = load(path)
            name = path.rsplit("/", 1)[-1].replace(".csv", "")
            for trace in make_3d_traces(df, name):
                fig.add_trace(trace)
        fig.update_layout(
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
            title="Source-space colony — " + ", ".join(paths),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
        )
    else:
        montage_pos = get_montage_positions()
        for path in paths:
            df, _ = load(path)
            name = path.rsplit("/", 1)[-1].replace(".csv", "")
            for trace in make_2d_traces(df, name, montage_pos):
                fig.add_trace(trace)
        add_head_outline(fig)
        fig.update_layout(
            title=f"Electrode colony ({mode}) — " + ", ".join(paths),
            xaxis=dict(scaleanchor="y", showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
            plot_bgcolor="white",
        )

    output_name = f"colony_viewer_{mode}.html"
    fig.write_html(output_name, include_plotlyjs=True)
    fig.show()
    print(f"saved {output_name}")
