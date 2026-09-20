from mne.minimum_norm import prepare_inverse_operator
import numpy as np
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
import sys
from pathlib import Path
from mne.datasets import sample
import mne
import os
from pnpl.datasets import LibriBrainWord

from colony import compute_gain, setup_inverse
from core import BANDS

SFREQ = 250
TMIN, TMAX = 0.0, 10.0
DATA_PATH = Path(".") / "pnpl" / "libribrain_word"
TRAIN_RUNS = [("0", str(s), "Sherlock1", "1") for s in range(1, 5)]   # sessions 1-4
VAL_RUNS   = [("0", "5", "Sherlock1", "1")]                            # session 5
TEST_RUNS  = [("0", "6", "Sherlock1", "1")] 

one_run = LibriBrainWord(
    data_path=str(DATA_PATH),
    include_run_keys=[TRAIN_RUNS[0]],
    #tmin=TMIN,
    #tmax=TMAX,
    standardize=False,     # show raw-ish values for this first look
    include_info=True,     # also return a dict with the word string, onset, etc.
    preload_files=False,   # download lazily instead of all-at-once
) 

meg_info = mne.channels.read_meg_canonical_info('neuromag')

with meg_info._unlock():
    meg_info['sfreq'] = SFREQ # libri serialized sfreq

meg, label_id, run_info = one_run[0]
print(meg.shape)

raw = mne.io.RawArray(meg, meg_info)
raw.info['dev_head_t'] = mne.transforms.Transform('meg', 'head', np.eye(4))

inv, src, bem = setup_inverse("pnpl-libriword", "subj0", None, ad_hoc_resting=True)
snr = 3.0
lambda2 = 1.0 / (snr ** 2)
prepared_inv = prepare_inverse_operator(
    inv,
    nave=1,
    lambda2=lambda2
)

src_data = mne.read_source_spaces(src)

for band_name, band in BANDS.items():
    low = band["low"]
    high = min(band["high"], SFREQ / 2.0 - 1)

    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4)

    # pass in a generator?!!
    # need to annotate with labels
    new_colonies = compute_gain(prepared_inv, raw_filtered,
        lambda2, 50 / 1000, None, include_vol=True, include_csd=True, include_inverse=True, 
        include_pos=True, include_neg=True)
    
    # 306 for every subject, we can use vol, csd, AND inverse

    for (source, group), new_colony in new_colonies.items():
        reg_colonies[(band_name, group)] = new_colony

sys.exit()

def model(meg: np.ndarray):
    return [], []

holdout = LibriBrainCompetitionHoldout(track="deep")   # "deep" or "broad"

print(holdout._file_paths)

primary, moses = [], []
for meg, _meta in holdout.iter_windows(batch_size=None):  # meg: (B, 306, 250)
    p, m = model(meg)                               # each (B, 50) probabilities
    primary.append(p); moses.append(m)

write_submission(
    "submission.csv",
    indices=holdout.indices,                 # required row order
    primary_probs=np.concatenate(primary),   # (N, 50) competition vocab, scored
    secondary_probs=np.concatenate(moses),   # (N, 50) Moses-50, optional/secondary
)
