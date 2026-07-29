import sqlite3
import events
import psutil
import time
from pathlib import Path

import mne
import scipy
import numpy as np
import matplotlib.pyplot as plt

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

OUT = "cta/"
BASE_PATH = Path("./physionet-motorimagery")
RECORDS = []

BANDS = {
    "whole": {
        "low": 1,
        "high": 100,
        "timesteps": []
    },
    "theta": {
        "low": 1,
        "high": 4,
        "timesteps": []
    }, 
    "delta": {
        "low": 4,
        "high": 8,
        "timesteps": []
    },
    "alpha": {
        "low": 8,
        "high": 13,
        "timesteps": [5, 10, 20, 30, 40]
    },
    "beta": {
        "low": 13,
        "high": 30,
        "timesteps": []
    },
    "gamma": {
        "low": 30,
        "high": 100,
        "timesteps": []
    },
}

with open(BASE_PATH / "RECORDS") as f:
    for line in f:
        RECORDS.append(line.strip())

def get_subject_data(record):
    subject = record[5:9]
    session = record[9:12]
    raw = mne.io.read_raw_edf(BASE_PATH / subject / f"{subject}{session}.edf", preload=True)
    raw.notch_filter(freqs=60.0, fir_design='firwin')
    events, _ = mne.events_from_annotations(raw)
    
    return raw, events

def get_raw_band_deviations(raw, bands, plot=False, band_data=BANDS):
    devs = []
    nyq = raw.info["sfreq"] / 2.0
    
    for band_name in bands:
        band = band_data[band_name]
        high = band["high"]
        
        if band["low"] > nyq:
            devs.append((band, None))
            continue
        if high > nyq:
            high = nyq - 1
        
        passed = raw.copy().filter(band["low"], high, fir_design="firwin")
        a = passed.copy().apply_hilbert(envelope=True).get_data()
        b = scipy.ndimage.uniform_filter1d(a, size=int(10.0 * passed.info["sfreq"]), axis=1)
        deviation = (a - b) / b
        devs.append((band_name, deviation))
    
        if plot:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title(f"Deviation at band {band['low']}-{band['high']} Hz")

            for i, y_items in enumerate(deviation):
                ax.plot(range(len(y_items)), y_items, label=f'Group {i+1}')

            ax.legend()
            plt.show()
    
    return devs


class SpikeStore:
    def __init__(self, db_path: str, batch_size: int = 5000):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.batch_size = batch_size
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id INTEGER NOT NULL,
                electrode TEXT NOT NULL,
                frequency INTEGER NOT NULL DEFAULT 0,
                jump_sum REAL NOT NULL DEFAULT 0.0,
                UNIQUE(parent_id, electrode)
            );

            CREATE TABLE IF NOT EXISTS counters (
                node_id INTEGER NOT NULL,
                event_name TEXT NOT NULL,
                frequency INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(node_id, event_name),
                FOREIGN KEY(node_id) REFERENCES nodes(id)
            );
            """
        )
        self.node_id_cache: dict[tuple[int, str], int] = {}
        self.pending_node_updates: dict[int, list[float]] = {}
        self.pending_counter_updates: dict[int, int] = {}
        self.pending_ops = 0

    def get_or_create_node_id(self, parent_id: int, electrode: str) -> int:
        key = (parent_id, electrode)
        cached = self.node_id_cache.get(key)
        
        if cached is not None:
            return cached

        node_id = self.conn.execute(
            """
            INSERT INTO nodes(parent_id, electrode)
            VALUES(?, ?)
            ON CONFLICT(parent_id, electrode) DO UPDATE SET id = id
            RETURNING id
            """,
            (parent_id, electrode),
        ).fetchone()[0]
        self.node_id_cache[key] = node_id
        
        return node_id

    def incr_spike(self, node_id: int, jump: float):
        node_update = self.pending_node_updates.get(node_id)
        if node_update is None:
            self.pending_node_updates[node_id] = [1.0, jump]
        else:
            node_update[0] += 1.0
            node_update[1] += jump

        self.pending_counter_updates[node_id] = self.pending_counter_updates.get(node_id, 0) + 1
        self.pending_ops += 1

        if self.pending_ops >= self.batch_size:
            self.flush()

    def close(self):
        self.flush()
        self.conn.commit()
        self.conn.close()

    def flush(self):
        if self.pending_ops == 0:
            return

        self.conn.executemany(
            """
            UPDATE nodes
            SET frequency = frequency + ?,
                jump_sum = jump_sum + ?
            WHERE id = ?
            """,
            [
                (int(update[0]), update[1], node_id)
                for node_id, update in self.pending_node_updates.items()
            ],
        )
        self.conn.executemany(
            """
            INSERT INTO counters(node_id, event_name, frequency)
            VALUES(?, 'spike', ?)
            ON CONFLICT(node_id, event_name)
            DO UPDATE SET frequency = frequency + excluded.frequency
            """,
            [
                (node_id, freq)
                for node_id, freq in self.pending_counter_updates.items()
            ],
        )

        self.pending_node_updates.clear()
        self.pending_counter_updates.clear()
        self.pending_ops = 0
        self.conn.commit()


def voltage_current_trains():
    def is_spike(old, new):
        return old >= new * 1.3
    
    with Live(Panel("Initializing...", expand=False), refresh_per_second=4) as live:
        target_records = RECORDS[10:11]
        for i, record in enumerate(target_records):
            raw, events = get_subject_data(record)
            sfreq = int(raw.info["sfreq"])
            deviations = get_raw_band_deviations(raw, ["alpha"], plot=True)

            # welp, we gotta do segmentation here too... at least the electrode field 
            # is swappable; just make a square grid for the raw electrodes and csd; 
            # hell, could we use the same logic, just different is_spike impls?
                # either downsegment to the lowest size eeg cap num so everything is consistent
                    # or... try and do nearest/most similar path matching... somehow

            # "If you use one fixed timespan across all bands, you'll be 
            # over-smoothing the fast bands or under-resolving the slow ones."

            # the plan: keep track of every leaf node and its prefix path; 
            # if a new spike is found, add it to every leaf node AND the root
            
            process = psutil.Process(os.getpid())
            timer = time.time()
            
            def update_live(bi, ti, chi, ls):
                updated_text = \
f"""Analyzing record {i+1}/{len(target_records)}
Band {bi}/{len(deviations)} ({band}) | Timestep {ti+1}/{len(BANDS[band]["timesteps"])} ({BANDS[band]["timesteps"][ti]}) | Channel {chi+1}/{len(raw.ch_names)}
Spike Branches: {ls}
CPU Load: {psutil.cpu_percent():.2f}%
RAM Usage: {process.memory_info().rss / (1024 ** 2):.2f}MB
Runtime: {time.time() - timer:.2f}s"""
                live.update(Panel(updated_text, title="Voltage Current Train Analysis", expand=False))
            
            bi = 0
            for band, deviation in deviations:
                bi += 1
                for ti, timestep in enumerate(BANDS[band]["timesteps"]):
                    update_live(bi, ti, 0, 0)
                    
                    db_path = f"{OUT}/{record}/{band}/ts{timestep}/spike_trie.sqlite"
                    store = SpikeStore(db_path)
                    root_id = store.get_or_create_node_id(-1, "root")
                    
                    last_spikes: list[int] = [root_id]
                    timestep_hop = int(np.ceil((timestep*1000) / sfreq))
                    
                    tested = set()
                    
                    for t in range(1, raw.n_times * sfreq * 1000, timestep):
                        current_spikes: list[int] = []
                        timestep_index = int(max(1, np.floor(t / sfreq * 1000)))
                        
                        if timestep_index in tested:
                            continue
                        else:
                            tested.add(timestep_index)
                
                        for ch_index, ch in enumerate(raw.ch_names):
                            old = deviation[ch_index][timestep_index - timestep_hop]
                            new = np.mean(deviation[ch_index][timestep_index - timestep_hop+1:timestep_index])
                            
                            if is_spike(old, new):
                                for parent_id in last_spikes:
                                    node_id = store.get_or_create_node_id(parent_id, ch)
                                    store.incr_spike(node_id, float(new - old))
                                    current_spikes.append(node_id)
                            
                            update_live(bi, ti, ch_index, len(last_spikes))
                    
                        last_spikes = [root_id] + current_spikes

                    store.close()


if __name__ == "__main__":
    voltage_current_trains()
