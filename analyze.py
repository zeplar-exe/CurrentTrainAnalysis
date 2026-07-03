from collections import Counter
from dataclasses import dataclass
import json

import mne
import scipy
import matplotlib.pyplot as plt

BASE_PATH = "./physionet-motorimagery"
RECORDS = []

with open(f"{BASE_PATH}/RECORDS") as f:
    for line in f:
        RECORDS.append(line.strip())

def get_subject_data(record):
    subject = record[5:9]
    session = record[9:12]
    raw = mne.io.read_raw_edf(f"{BASE_PATH}/{subject}/{subject}{session}.edf", preload=True)
    events = mne.events_from_annotations(raw)
    
    return raw, events

def get_raw_band_deviations(raw, plot=False):
    devs = []
    
    for band in [(0,4), (4,8),(8,12),(12,30), (30, min(nyq, 100) - 0.5)]:
        passed = raw.copy().filter(band[0], band[1], fir_design="firwin")
        a = passed.copy().apply_hilbert(envelope=True).get_data()
        b = scipy.ndimage.uniform_filter1d(a, size=int(2.0 * passed.info["sfreq"]), axis=1)
        deviation = a - b
        devs.append((band, deviation))
    
        if plot:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title(f"Deviation at band {band[0]}-{band[1]} Hz")

            for i, y_items in enumerate(deviation):
                ax.plot(range(len(y_items)), y_items, label=f'Group {i+1}')

            ax.legend()
            plt.show()
    
    return devs

def is_spike(old, new):
    return old >= new * 1.3


@dataclass
class SpikeNode:
    electrode: str
    frequency: int
    events: Counter
    children: list[SpikeNode]
    
    def incr(self):
        self.frequency += 1

for record in RECORDS[10:11]:
    raw, events = get_subject_data(record)
    nyq = raw.info["sfreq"] / 2.0
    deviations = get_raw_band_deviations(raw)


    TIMESTEPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # welp, we gotta do segmentation here too... at least the electrode field 
    # is swappable; just make a square grid for the raw electrodes and csd; 
    # hell, could we use the same logic, just different is_spike impls?
        # either downsegment to the lowest size eeg cap num so everything is consistent
            # or... try and do nearest/most similar path matching... somehow

    # "If you use one fixed timespan across all bands, you'll be 
    # over-smoothing the fast bands or under-resolving the slow ones."

    # it would be helpful to make sure things don't explude by making sure that 
    # the number of spikes per timestep isn't literally *every* electrode

    # the plan: keep track of every leaf node and its prefix path; 
    # if a new spike is found, add it to every leaf node AND the root

    for band, deviation in deviations:
        for timestep in TIMESTEPS:
            root = SpikeNode("root", 0, Counter(), [])
            leaves: list[tuple[list[str], SpikeNode]] = []
            
            def traverse(path: list[str]) -> tuple[SpikeNode, bool]:
                current = root
                created = False
                
                for electrode in path:
                    found = None
                    for child in current.children:
                        if child.electrode == electrode:
                            found = child
                            break
                    
                    if found is None:
                        new_node = SpikeNode(electrode, 0, Counter(), [])
                        current.children.append(new_node)
                        current = new_node
                        created = True
                    else:
                        current = found
                
                return current, created
            
            for t in range(1, raw.n_times, timestep):
                new_leaves = []
                
                for ch_index, ch in enumerate(raw.ch_names):
                    old = deviation[ch_index][t - 1]
                    new = deviation[ch_index][t]
                    
                    if is_spike(old, new):
                        s, created = traverse([ch])
                        s.incr()
                        if created:
                            new_leaves.append(s)
                        root.children.append(s)
                        
                        for path, leaf in leaves:
                            s2, created = traverse(path + [ch])
                            s2.incr()
                            if created:
                                new_leaves.append(s2)
                            leaf.children.append(s2)
                            # ah shit, so this basically auto-terminates the restarting paths... how the FUCK do we make this a suffix array or whatever...
            
                leaves = new_leaves + [l for l in leaves if not l.children]
        
            with open(f"{OUT}/{record}/{band}/ts{timestep}/spike_trie.json", "w") as f:
                json.dump(root, f, indent=1)
