"""
3D viewer for source-space total gain.
Expects CSV with columns: x, y, z, gain
Multiple CSVs (per-band) supported.

Usage:
    python view_gain.py alpha.csv
    python view_gain.py alpha.csv beta.csv theta.csv
"""
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

def load(path):
    df = pd.read_csv(path)
    # flexible column matching — takes first 3 numeric as xyz, last as gain
    if all(c in df.columns for c in ['x', 'y', 'z', 'gain']):
        return df
    nums = df.select_dtypes('number').columns
    df = df[nums].copy()
    df.columns = ['x', 'y', 'z', 'gain']
    return df

def make_trace(df, name, colorscale='Hot'):
    return go.Scatter3d(
        x=df.x, y=df.y, z=df.z,
        mode='markers',
        marker=dict(
            size=3,
            color=df.gain,
            colorscale=colorscale,
            cmin=0,
            cmax=df.gain.quantile(0.99),   # clip outliers for color range
            colorbar=dict(title=name),
            opacity=0.8,
        ),
        name=name,
        hovertemplate='x:%{x:.1f} y:%{y:.1f} z:%{z:.1f}<br>gain:%{marker.color:.3f}<extra></extra>',
    )

scales = ['Hot', 'Viridis', 'Plasma', 'Inferno', 'Cividis']

fig = go.Figure()
for i, path in enumerate(sys.argv[1:]):
    df = load(path)
    name = Path(path).stem
    fig.add_trace(make_trace(df, name, scales[i % len(scales)]))

fig.update_layout(
    scene=dict(
        xaxis_title='x', yaxis_title='y', zaxis_title='z',
        aspectmode='data',     # preserve real proportions
    ),
    title='Source-space total gain',
    margin=dict(l=0, r=0, t=40, b=0),
)

# toggle bands on/off via legend clicks
fig.update_layout(legend=dict(itemclick='toggle', itemdoubleclick='toggleothers'))
fig.write_html('gain_viewer.html', include_plotlyjs=True)
fig.show()
print('saved gain_viewer.html')