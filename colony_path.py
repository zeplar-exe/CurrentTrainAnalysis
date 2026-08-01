"""
Thanks Claude
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from colony import DATASET_SPECS, BANDS, Colony

GROWTH_ROOT = Path("growth")
COALESCE_ROOT = Path("coalesce")
GROWTH_INDEX = {"raw": 0, "pos": 1, "neg": 2}

presets = {
    "grasplift_handstart": {
        "dataset": "grasplift",
        "subjects": None,
        "event": "HandStart",
        "coalesce_event": "HandStart",
        "band": "standard",
        "timestep": 50,
        "cluster_window": 5,
        "growth_type": "pos+neg",
    },
    "grasplift_liftoff": {
        "dataset": "grasplift",
        "subjects": None,
        "event": "LiftOff",
        "coalesce_event": "LiftOff",
        "band": "standard",
        "timestep": 50,
        "cluster_window": 5,
        "growth_type": "pos+neg",
    },
}


class Stereotype:
    def __init__(self, weights: np.ndarray, cluster_window: int):
        self.weights = weights
        self.cluster_window = cluster_window
        self._sum = np.zeros((len(weights), cluster_window))
        self._count = 0

    def fit(self, window: np.ndarray):
        self._sum += window * self.weights[:, np.newaxis]
        self._count += 1

    def path(self):
        if self._count == 0:
            return None
        return self._sum / self._count


def load_weights(dataset, band, event, sign):
    path = COALESCE_ROOT / dataset / "inverse" / band / event / f"{sign}.csv"
    df = pd.read_csv(path)
    values: np.ndarray = df["value"].values
    if sign == "neg":
        w = np.abs(values - np.max(values))
        p99 = np.percentile(w, 99)
        return w / p99 if p99 > 1e-6 else w
    w = values - np.min(values)
    p99 = np.percentile(w, 99)
    return w / p99 if p99 > 1e-6 else w


def load_coordinates(dataset):
    dataset_dir = GROWTH_ROOT / dataset
    for subj_dir in sorted(dataset_dir.iterdir()):
        coord_path = subj_dir / "coordinates.npy"
        if coord_path.exists():
            return np.load(coord_path)
    raise FileNotFoundError(f"No coordinates.npy found in {dataset_dir}")


def build_path(preset_name):
    cfg = presets[preset_name]
    spec = DATASET_SPECS[cfg["dataset"]]
    subjects = cfg["subjects"] or spec["subjects"]
    types = cfg["growth_type"].split("+")
    coalesce_event = cfg.get("coalesce_event", cfg["event"])

    coordinates = load_coordinates(cfg["dataset"])

    cw = cfg["cluster_window"]

    band = cfg["band"]

    stereos = {}
    for t in types:
        sign = t if t in ("pos", "neg") else "pos"
        weights = load_weights(cfg["dataset"], band, coalesce_event, sign)
        stereos[t] = Stereotype(weights, cw)

    for subject in subjects:
        for t in types:
            event_dir = (GROWTH_ROOT / cfg["dataset"] / subject / band
                         / f"t{cfg['timestep']}c{cw}" / cfg["event"])
            if not event_dir.exists():
                continue
            for record_path in sorted(event_dir.glob("record_*.npy")):
                data = np.load(record_path)
                growth = data[GROWTH_INDEX[t]]
                for w in range(growth.shape[0]):
                    stereos[t].fit(growth[w])

    paths = {t: stereos[t].path() for t in types}

    interp_factor = cfg["timestep"] // 5
    if interp_factor > 1:
        from scipy.interpolate import interp1d
        for t in paths:
            p = paths[t]
            n_vert, n_orig = p.shape
            x_orig = np.arange(n_orig)
            x_new = np.linspace(0, n_orig - 1, n_orig * interp_factor)
            f = interp1d(x_orig, p, axis=1, kind="cubic")
            paths[t] = f(x_new)

    n_frames = next(iter(paths.values())).shape[1]
    ms_per_frame = 5

    return coordinates, paths, cfg, n_frames, ms_per_frame


def create_animation(preset_name):
    coordinates, paths, cfg, n_frames, ms_per_frame = build_path(preset_name)
    types = list(paths.keys())
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    frame_indices = list(range(n_frames))

    trace_defs = []
    for t in types:
        p = paths[t]
        if t == "neg":
            p = np.abs(p)
        vmax = np.percentile(np.abs(p), 99) or 1.0
        cscale = "Reds" if t == "pos" else ("Blues" if t == "neg" else "RdBu_r")
        trace_defs.append((t, p, cscale, 0, vmax))

    BASE_SIZE = 1.5
    MAX_SIZE = 15

    frames = []
    for fi in frame_indices:
        frame_data = []
        for name, data, cscale, cmin, cmax in trace_defs:
            vals = data[:, fi]
            magnitudes = np.abs(vals) / (cmax or 1)
            sizes = BASE_SIZE + magnitudes * (MAX_SIZE - BASE_SIZE)
            frame_data.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=sizes.tolist(),
                    color=vals.tolist(),
                    colorscale=cscale, cmin=cmin, cmax=cmax,
                    opacity=0.7,
                    colorbar=dict(title=name),
                    line=dict(width=0),
                ),
                name=name,
                showlegend=True,
            ))
        frames.append(go.Frame(data=frame_data, name=str(fi)))

    slider_steps = [
        dict(method="animate",
             args=[[str(fi)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
             label=f"{fi * ms_per_frame}ms")
        for fi in frame_indices
    ]

    toggle_buttons = []
    for i, (name, *_) in enumerate(trace_defs):
        toggle_buttons.append(
            dict(label=f"{name} on", method="restyle",
                 args=[{"visible": [True]}, [i]]))
        toggle_buttons.append(
            dict(label=f"{name} off", method="restyle",
                 args=[{"visible": ["legendonly"]}, [i]]))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=f"Colony Path: {preset_name}",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
            updatemenus=[
                dict(type="buttons", showactive=False, y=0, x=0.5, xanchor="center",
                     buttons=[
                         dict(label="Play", method="animate",
                              args=[None, {"frame": {"duration": 80, "redraw": True},
                                           "fromcurrent": True, "transition": {"duration": 0}}]),
                         dict(label="Pause", method="animate",
                              args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                     ]),
                dict(type="buttons", showactive=True, y=1.0, x=0.0, xanchor="left",
                     direction="right", buttons=toggle_buttons),
            ],
            sliders=[dict(active=0, steps=slider_steps, x=0.1, len=0.8,
                          currentvalue=dict(prefix="Time: "))],
        ),
    )

    output = f"colony_path_{preset_name}.html"
    fig.write_html(output)
    fig.show()
    print(f"Saved {output}")
    return fig


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else list(presets.keys())[0]
    if name not in presets:
        print(f"Unknown preset '{name}'. Available: {', '.join(presets.keys())}")
        sys.exit(1)
    create_animation(name)
