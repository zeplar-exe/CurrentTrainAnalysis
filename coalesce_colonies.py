import pandas as pd
import sys
from pathlib import Path
from typing import List


def coalesce(paths: List[str], output_path: str):
    dfs = [pd.read_csv(p) for p in paths]
    keys = dfs[0].columns
    for i, df in enumerate(dfs[1:], 1):
        if not keys.equals(df.columns):
            raise ValueError(f"Column mismatch: {paths[0]} has {list(keys)}, {paths[i]} has {list(df.columns)}")

    key_cols = [c for c in keys if c in ("x", "y", "z", "electrode")]
    value_cols = [c for c in keys if c not in key_cols]

    combined = pd.concat(dfs, ignore_index=True)
    result = combined.groupby(key_cols, as_index=False)[value_cols].sum()
    result.to_csv(output_path, index=False)


def coalesce_event_band(dataset, event, band, vol=True, csd=True, inverse=True):
    colonies_root = Path("colonies") / dataset
    output_root = Path("coalesce") / dataset
    subjects = sorted(p.name for p in colonies_root.iterdir() if p.is_dir() and not p.name.startswith("."))

    for method, enabled in [("vol", vol), ("csd", csd), ("inverse", inverse)]:
        if not enabled:
            continue
        for sign in ("pos", "neg"):
            paths = []
            for subject in subjects:
                csv_path = colonies_root / subject / sign / method / band / f"{event}.csv"
                if csv_path.exists():
                    paths.append(str(csv_path))

            if len(paths) == 0:
                continue

            out_dir = output_root / method / band / event
            out_dir.mkdir(parents=True, exist_ok=True)
            coalesce(paths, str(out_dir / f"{sign}.csv"))


EVENTS = [
    "task1_real_left_fist", "task1_real_right_fist",
    "task2_imagine_left_fist", "task2_imagine_right_fist",
    "task3_real_both_fists", "task3_real_both_feet",
    "task4_imagine_both_fists", "task4_imagine_both_feet",
]
BANDS = ["whole", "standard", "alpha", "beta", "delta", "theta", "gamma"]

if __name__ == "__main__":
    for event in EVENTS:
        for band in BANDS:
            coalesce_event_band("eegmmidb", event, band)
