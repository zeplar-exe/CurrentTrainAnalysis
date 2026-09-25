# Word durations for LibriBrain. The aligner's durations ship with the dataset:
# every *_events.tsv has a `duration` column on rows with kind == "word".
# Joined on `timeds`, the 250 Hz timebase the serialised h5 uses.
# > Thanks Claude

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

SFREQ = 250
STATS_PATH = Path("word_duration_stats.json")
ALL_RUNS = [("0", str(s), "Sherlock1", "1") for s in range(1, 10 + 1)]


def _events_path(data_path, run_key):
    subject, session, task, run = run_key
    return (
        Path(data_path)
        / "Sherlock1"
        / "derivatives"
        / "events"
        / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
    )


def _words(data_path, run_key):
    df = pd.read_csv(_events_path(data_path, run_key), sep="\t")
    words = df[df["kind"] == "word"].copy()
    words["word"] = words["segment"].astype(str).str.strip().str.lower().str.replace(
        "’", "'", regex=False
    )
    return words.dropna(subset=["timeds", "duration"])


@lru_cache(maxsize=None)
def _durations(data_path, run_key):
    words = _words(data_path, run_key)
    return {
        int(round(t * SFREQ)): float(d)
        for t, d in zip(words["timeds"], words["duration"])
    }


def word_duration(data_path, run_key, onset):
    """Duration in seconds of the word at `onset` (seconds) in `run_key`."""
    table = _durations(str(data_path), tuple(run_key))
    sample = int(round(float(onset) * SFREQ))
    for offset in (0, -1, 1):
        if sample + offset in table:
            return table[sample + offset]
    raise KeyError(f"no word at onset {onset}s in run {run_key}")


def build_stats(data_path, run_keys=ALL_RUNS, out_path=STATS_PATH):
    """Mean + SD duration per word across every run, written to JSON.

    Durations come from the aligner, not the labels, so there's no reason to
    hold out sessions. Runs whose events TSV isn't downloaded yet are skipped.
    """
    frames = []
    for run_key in run_keys:
        if _events_path(data_path, run_key).exists():
            frames.append(_words(data_path, run_key))
    if not frames:
        raise FileNotFoundError(f"no events TSVs on disk under {data_path}")

    words = pd.concat(frames, ignore_index=True)
    agg = words.groupby("word")["duration"].agg(["count", "mean", "std"])
    agg["std"] = agg["std"].fillna(0.0)  # single-occurrence words

    stats = {
        word: {
            "count": int(row["count"]),
            "mean": round(float(row["mean"]), 4),
            "sd": round(float(row["std"]), 4),
        }
        for word, row in agg.iterrows()
    }
    Path(out_path).write_text(json.dumps(stats, indent=1, sort_keys=True))
    return stats


@lru_cache(maxsize=None)
def load_stats(path=STATS_PATH):
    """word -> {count, mean, sd}, built by build_stats."""
    return json.loads(Path(path).read_text())


def word_mean_duration(word, path=STATS_PATH, default=None):
    """Mean training duration for `word`, or `default` if it never occurs."""
    entry = load_stats(path).get(str(word).strip().lower().replace("’", "'"))
    return default if entry is None else entry["mean"]

def word_sd_duration(word, path=STATS_PATH, default=None):
    """Mean training duration for `word`, or `default` if it never occurs."""
    entry = load_stats(path).get(str(word).strip().lower().replace("’", "'"))
    return default if entry is None else entry["sd"]


if __name__ == "__main__":
    stats = build_stats(Path(".") / "pnpl" / "libribrain_word")
    print(f"{len(stats)} words -> {STATS_PATH}")
