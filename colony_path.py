from pathlib import Path
import sys
import numpy as np
import plotly.graph_objects as go
from colony import DATASET_SPECS, BANDS

GROWTH_ROOT = Path("growth")

GROWTH_INDEX = {"raw": 0, "pos": 1, "neg": 2}

presets = {
    "grasplift_handstart": {
        "dataset": "grasplift",
        "subjects": None,
        "event": "HandStart",
        "bands": ["alpha", "beta"],
        "timestep": 50,
        "cluster_window": 5,
        "growth_type": "pos+neg",
    },
    "grasplift_liftoff": {
        "dataset": "grasplift",
        "subjects": None,
        "event": "LiftOff",
        "bands": ["alpha", "beta"],
        "timestep": 50,
        "cluster_window": 5,
        "growth_type": "pos+neg",
    },
}


def load_coordinates(dataset):
    dataset_dir = GROWTH_ROOT / dataset
    for subj_dir in sorted(dataset_dir.iterdir()):
        coord_path = subj_dir / "coordinates.npy"
        if coord_path.exists():
            return np.load(coord_path)
    raise FileNotFoundError(f"No coordinates.npy found in {dataset_dir}")


def load_growth(preset_name):
    cfg = presets[preset_name]
    spec = DATASET_SPECS[cfg["dataset"]]
    subjects = cfg["subjects"] or spec["subjects"]
    types = cfg["growth_type"].split("+")

    coordinates = load_coordinates(cfg["dataset"])

    per_type = {t: [] for t in types}

    for subject in subjects:
        for band in cfg["bands"]:
            event_dir = (GROWTH_ROOT / cfg["dataset"] / subject / band
                         / f"t{cfg['timestep']}c{cfg['cluster_window']}" / cfg["event"])
            if not event_dir.exists():
                continue
            for record_path in sorted(event_dir.glob("record_*.npy")):
                data = np.load(record_path)
                for t in types:
                    per_type[t].append(data[GROWTH_INDEX[t]])

    if all(len(v) == 0 for v in per_type.values()):
        raise ValueError(f"No data found for preset '{preset_name}'")

    result = {}
    for t, entries in per_type.items():
        min_win = min(e.shape[0] for e in entries)
        stacked = np.stack([e[:min_win] for e in entries])
        averaged = stacked.mean(axis=0)
        window_means = averaged.mean(axis=2)
        cumulative = np.cumsum(window_means, axis=0)
        result[t] = cumulative

    return coordinates, result, cfg


def create_animation(preset_name, max_frames=200):
    coordinates, growth, cfg = load_growth(preset_name)
    types = list(growth.keys())
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    n_frames = growth[types[0]].shape[0]
    step = max(1, n_frames // max_frames)
    frame_indices = list(range(0, n_frames, step))

    ms_per_window = cfg["timestep"] * cfg["cluster_window"]

    traces = []
    if len(types) == 2 and "pos" in types and "neg" in types:
        pos_data = growth["pos"]
        neg_data = np.abs(growth["neg"])
        pos_max = np.percentile(pos_data, 99) or 1.0
        neg_max = np.percentile(neg_data, 99) or 1.0
        traces.append(("pos (ERS)", pos_data, "Reds", 0, pos_max))
        traces.append(("neg (ERD)", neg_data, "Blues", 0, neg_max))
    else:
        t = types[0]
        data = growth[t]
        vmax = np.percentile(np.abs(data), 99) or 1.0
        if t == "raw":
            traces.append((t, data, "RdBu_r", -vmax, vmax))
        elif t == "pos":
            traces.append((t, data, "Reds", 0, vmax))
        else:
            traces.append((t, np.abs(data), "Blues", 0, vmax))

    n_traces = len(traces)
    frames = []
    for fi in frame_indices:
        frame_data = []
        for ti, (name, data, cscale, cmin, cmax) in enumerate(traces):
            vals = data[fi]
            sizes = np.clip(np.abs(vals) / (cmax or 1) * 4 + 0.5, 0.5, 6)
            cbar = dict(title=name, len=0.3, y=0.8 - ti * 0.5) if n_traces > 1 else dict(title="Growth")
            frame_data.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=sizes.tolist(), color=vals.tolist(),
                    colorscale=cscale, cmin=cmin, cmax=cmax,
                    opacity=0.6, colorbar=cbar,
                ),
                name=name,
            ))
        frames.append(go.Frame(data=frame_data, name=str(fi)))

    slider_steps = [
        dict(method="animate",
             args=[[str(fi)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
             label=f"{fi * ms_per_window}ms")
        for fi in frame_indices
    ]

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=f"Colony Path: {preset_name}",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
            updatemenus=[dict(
                type="buttons", showactive=False, y=0, x=0.5, xanchor="center",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, {"frame": {"duration": 80, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(active=0, steps=slider_steps, x=0.1, len=0.8,
                          currentvalue=dict(prefix="Time: "))],
        ),
    )

    output = f"colony_path_{preset_name}.html"
    fig.write_html(output)
    print(f"Saved {output}")
    return fig


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else list(presets.keys())[0]
    if name not in presets:
        print(f"Unknown preset '{name}'. Available: {', '.join(presets.keys())}")
        sys.exit(1)
    create_animation(name)
