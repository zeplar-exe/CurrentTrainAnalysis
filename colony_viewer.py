"""
3D viewer for source-space total gain with percentile highlighting.
Top 5% = red, 5-10% = orange, 10-15% = yellow, rest = dim gray.
Each group is a separate toggleable trace.

Usage:
    python view_gain_pct.py alpha.csv
    python view_gain_pct.py alpha.csv beta.csv
"""
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def load(path):
    df = pd.read_csv(path)
    if all(c in df.columns for c in ['x', 'y', 'z', 'gain']):
        return df
    nums = df.select_dtypes('number').columns
    df = df[nums].copy()
    df.columns = ['x', 'y', 'z', 'gain']
    return df

def make_percentile_traces(df, name):
    p85 = df.gain.quantile(0.85)
    p90 = df.gain.quantile(0.90)
    p95 = df.gain.quantile(0.95)

    rest   = df[df.gain < p85]
    top_15 = df[(df.gain >= p85) & (df.gain < p90)]
    top_10 = df[(df.gain >= p90) & (df.gain < p95)]
    top_5  = df[df.gain >= p95]

    hover = 'x:%{x:.1f} y:%{y:.1f} z:%{z:.1f}<br>gain:%{customdata:.3f}<extra></extra>'

    traces = []

    # bottom 85% — dim gray
    traces.append(go.Scatter3d(
        x=rest.x, y=rest.y, z=rest.z,
        mode='markers',
        marker=dict(size=2, color='#9ca3af', opacity=0.15),
        customdata=rest.gain,
        name=f'{name} <85% (< {p85:.2f})',
        hovertemplate=hover,
    ))

    # top 10-15% — yellow
    traces.append(go.Scatter3d(
        x=top_15.x, y=top_15.y, z=top_15.z,
        mode='markers',
        marker=dict(size=3, color='#eab308', opacity=0.7),
        customdata=top_15.gain,
        name=f'{name} top 10-15% ({p85:.2f}-{p90:.2f})',
        hovertemplate=hover,
    ))

    # top 5-10% — orange
    traces.append(go.Scatter3d(
        x=top_10.x, y=top_10.y, z=top_10.z,
        mode='markers',
        marker=dict(size=4, color='#ea580c', opacity=0.85),
        customdata=top_10.gain,
        name=f'{name} top 5-10% ({p90:.2f}-{p95:.2f})',
        hovertemplate=hover,
    ))

    # top 5% — red
    traces.append(go.Scatter3d(
        x=top_5.x, y=top_5.y, z=top_5.z,
        mode='markers',
        marker=dict(size=5, color='#dc2626', opacity=0.95),
        customdata=top_5.gain,
        name=f'{name} top 5% (> {p95:.2f})',
        hovertemplate=hover,
    ))

    return traces

fig = go.Figure()
for path in sys.argv[1:]:
    df = load(path)
    name = path.rsplit('/', 1)[-1].replace('.csv', '')
    for trace in make_percentile_traces(df, name):
        fig.add_trace(trace)

fig.update_layout(
    scene=dict(
        xaxis_title='x', yaxis_title='y', zaxis_title='z',
        aspectmode='data',
    ),
    title='Source-space total gain — percentile groups',
    margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(itemclick='toggle', itemdoubleclick='toggleothers'),
)

fig.write_html('gain_viewer_pct.html', include_plotlyjs=True)
fig.show()
print('saved gain_viewer_pct.html')