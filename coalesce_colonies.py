import numpy as np
import pandas as pd
from pathlib import Path
from colony import Colony


def coalesce_colonies(colonies: list[Colony]) -> Colony:
    first = colonies[0]
    size = len(first.colony_pos) if first.include_pos else len(first.colony_neg) if first.include_neg else len(first.colony_raw) if first.include_raw else len(first.colony_abs)
    result = Colony(size, include_raw=first.include_raw, include_abs=first.include_abs, include_pos=first.include_pos, include_neg=first.include_neg)

    for colony in colonies:
        if result.include_pos:
            pos_99 = np.percentile(colony.colony_pos, 99)
            result.colony_pos += np.clip(colony.colony_pos / pos_99, None, 1.0) if pos_99 > 0 else colony.colony_pos
        if result.include_neg:
            neg_99 = np.percentile(np.abs(colony.colony_neg), 99)
            result.colony_neg += np.clip(colony.colony_neg / neg_99, -1.0, None) if neg_99 > 0 else colony.colony_neg
        if result.include_raw:
            raw_99 = np.percentile(np.abs(colony.colony_raw), 99)
            result.colony_raw += np.clip(colony.colony_raw / raw_99, -1.0, 1.0) if raw_99 > 0 else colony.colony_raw
        if result.include_abs:
            abs_99 = np.percentile(colony.colony_abs, 99)
            result.colony_abs += np.clip(colony.colony_abs / abs_99, None, 1.0) if abs_99 > 0 else colony.colony_abs

    return result


def _coalesce_csvs(paths: list[str], output_path: str):
    dfs = [pd.read_csv(p) for p in paths]
    keys = dfs[0].columns
    key_cols = [c for c in keys if c in ("x", "y", "z", "electrode")]
    value_cols = [c for c in keys if c not in key_cols]

    for i, df in enumerate(dfs[1:], 1):
        if not keys.equals(df.columns):
            raise ValueError(f"Column mismatch: {paths[0]} has {list(keys)}, {paths[i]} has {list(df.columns)}")
        df[value_cols] = (df[value_cols] / df[value_cols].quantile(0.99)).clip(upper=1.0)

    combined = pd.concat(dfs, ignore_index=True)
    result = combined.groupby(key_cols, as_index=False)[value_cols].sum()
    result.to_csv(output_path, index=False)


def coalesce_event_band(dataset, event, band, mirrored, vol=True, csd=True, inverse=True):
    colonies_root = Path("colonies") / dataset
    output_root = Path("coalesce") / dataset
    subjects = sorted(p.name for p in colonies_root.iterdir() if p.is_dir() and not p.name.startswith("."))

    for method, enabled in [("vol", vol), ("csd", csd), ("inverse", inverse)]:
        if not enabled:
            continue
        for sign in ("pos", "neg"):
            paths = []
            for subject in subjects:
                mirror_key = "regular" if not mirrored else "mirrored"
                csv_path = colonies_root / subject / sign / method / band / mirror_key / f"{event}.csv"
                if csv_path.exists():
                    paths.append(str(csv_path))

            if len(paths) == 0:
                continue

            out_dir = output_root / method / band / event
            out_dir.mkdir(parents=True, exist_ok=True)
            _coalesce_csvs(paths, str(out_dir / f"{sign}.csv"))


EVENTS = [
    "HandStart",
    "FirstDigitTouch",
    "LiftOff",
]
BANDS = ["whole", "standard", "alpha", "beta", "delta", "theta", "gamma"]

if __name__ == "__main__":
    for event in EVENTS:
        for band in BANDS:
            coalesce_event_band("grasplift", event, band, mirrored=False)
