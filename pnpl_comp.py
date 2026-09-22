from collections import defaultdict
from itertools import groupby
import pickle

from mne.minimum_norm import apply_inverse_raw, prepare_inverse_operator
import numpy as np
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
import sys
from pathlib import Path
from mne.datasets import sample
import mne
import torch
import os
from heapq import nlargest
from pnpl.datasets import LibriBrainWord
from pnpl.competition import load_vocabulary
from colony import MultiColony, compute_gain, setup_inverse
from colony_viewer import show_colony
from core import BANDS
from tqdm import tqdm
import mne
from pathlib import Path
import warnings
from skorch import NeuralNetClassifier
import torch.nn as nn

warnings.filterwarnings(
    action="ignore", 
    category=RuntimeWarning, 
    message=".*is longer than the signal.*"
)

mne.set_log_level('ERROR')

FIF = "~/.cache/huggingface/hub/datasets--pnpl--LibriBrain/snapshots/5a7c332b34fc7be329c4df3e527a41f67bfd878f/Sherlock1/sub-0/ses-1/meg/sub-0_ses-1_task-Sherlock1_run-1_meg.fif"

TMIN, TMAX = 0.0, 0.4 # 1.0
DATA_PATH = Path(".") / "pnpl" / "libribrain_word"
TRAIN_RUNS = [("0", str(s), "Sherlock1", "1") for s in range(1, 7 + 1)]   # sessions 1-7
VALIDATION_RUNS = [("0", str(s), "Sherlock1", "1") for s in range(8, 9 + 1)]   # sessions 8-9
TEST_RUNS  = [("0", str(s), "Sherlock1", "1") for s in range(10, 10 + 1)]  # session 10

PRIMARY_VOCAB = load_vocabulary("primary")      # the 50 competition words, in order
MOSES_VOCAB = load_vocabulary("moses")        # the 50 Moses words (secondary metric)

def normalize_word(w):
    return str(w).strip().lower().replace("’", "'")

PRIMARY_VOCAB_TO_ID = {normalize_word(w): i for i, w in enumerate(PRIMARY_VOCAB)}
MOSES_VOCAB_TO_ID = {normalize_word(w): i for i, w in enumerate(MOSES_VOCAB)}
PRIMARY_ID_TO_VOCAB = {i: normalize_word(w) for i, w in enumerate(PRIMARY_VOCAB)}
MOSES_ID_TO_VOCAB = {i: normalize_word(w) for i, w in enumerate(MOSES_VOCAB)}

SFREQ = 250 # libri serialized sfreq
TIMESTEP = 25 / 1000 # s
MULTICOLONY_STEP = 75 / 1000 # s

TARGET_BANDS = BANDS.copy()
del TARGET_BANDS["whole"]
del TARGET_BANDS["standard"]

PERCENTILE = 0.75

MODEL_STATE_PATH = Path("./pnpl/model_state.pkl")

info_fif = mne.io.read_info(FIF)
info_fif = mne.pick_info(info_fif, mne.pick_types(info_fif, meg=True, exclude=[]))
with info_fif._unlock():
    info_fif['sfreq'] = SFREQ

def create_raw(meg):
    raw = mne.io.RawArray(meg, info_fif, verbose='error')
    raw.info['dev_head_t'] = mne.transforms.Transform('meg', 'head', np.eye(4))
    
    return raw

inv, src, bem = setup_inverse("pnpl-libriword", "subj0", None, ad_hoc_resting=True, info=info_fif, root=Path("./pnpl/inverse"))
snr = 3.0
lambda2 = 1.0 / (snr ** 2)
prepared_inv = prepare_inverse_operator(
    inv,
    nave=1,
    lambda2=lambda2
)

src_data = mne.read_source_spaces(src)

primary_colonies_words: dict[tuple[str, str, str], MultiColony] = {}
moses_colonies_words: dict[tuple[str, str, str], MultiColony] = {}
primary_band_clfs = {}
moses_band_clfs = {}
label_distribution: dict[str, int] = defaultdict(int)

class WordCNN(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 2),
        )

    def forward(self, X):
        # X: (batch, n_vertices * n_bands, n_timepoints)
        return self.net(X)

def save_colony_state(f: str | Path = MODEL_STATE_PATH):
    path = Path(f)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "primary_colonies_words": primary_colonies_words,
        "moses_colonies_words": moses_colonies_words,
        "primary_band_clfs": primary_band_clfs,
        "moses_band_clfs": moses_band_clfs,
        "label_distribution": dict(label_distribution),
        "metadata": {
            "sfreq": SFREQ,
            "timestep": TIMESTEP,
            "multicolony_step": MULTICOLONY_STEP,
            "target_bands": list(TARGET_BANDS.keys()),
        },
    }

    with path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_colony_state(f: str | Path = MODEL_STATE_PATH):
    path = Path(f)
    with path.open("rb") as handle:
        state = pickle.load(handle)

    global primary_colonies_words, moses_colonies_words
    global primary_band_clfs, moses_band_clfs, label_distribution

    primary_colonies_words = state["primary_colonies_words"]
    moses_colonies_words = state["moses_colonies_words"]
    primary_band_clfs = state["primary_band_clfs"]
    moses_band_clfs = state["moses_band_clfs"]
    label_distribution = defaultdict(int, state.get("label_distribution", {}))

    return state

def _digest(raw: mne.io.RawArray, colony_container: dict[tuple[str, str, str], MultiColony], label: str):
    for band_name, band in TARGET_BANDS.items():
        low = band["low"]
        high = min(band["high"], SFREQ / 2.0 - 1)

        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4, verbose='error')

        new_colonies = compute_gain(prepared_inv, raw_filtered,
            lambda2, TIMESTEP, MULTICOLONY_STEP, None, 
            include_vol=True, include_csd=False, include_inverse=True, 
            include_pos=True, include_neg=False, use_epochs=False)

        for (source, _), new_colony in new_colonies.items():
            k = (source, band_name, label)
            if k in colony_container:
                colony_container[k].merge(new_colony)
            else:
                colony_container[k] = new_colony

def _train_clf(raw: mne.io.RawArray, colony_container: dict[tuple[str, str, str], MultiColony], model_container: dict[tuple[str, str], NeuralNetClassifier], label: str):
    label_distribution[label] += 1
    total = sum(label_distribution.values())
    pos_count = label_distribution[label]
    neg_count = total - pos_count
    neg_w = pos_count / total if total > 0 else 0.5
    pos_w = neg_count / total if total > 0 else 0.5

    band_grouped = groupby(sorted(colony_container.items(), key=lambda x: (x[0][0], x[0][2])), lambda x: (x[0][0], x[0][2]))
    for (source, lb), group in band_grouped:
        binary = lb == label
        b = []

        for (_, band_name, _), colony in group:
            weights = colony.pos_weights()
        
            band = TARGET_BANDS[band_name]
            low = band["low"]
            high = min(band["high"], SFREQ / 2.0 - 1)

            raw_filtered = raw.copy()
            raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4, verbose='error')
            
            if source == "vol":
                src = raw_filtered.get_data().astype(np.float32)
            elif source == "inverse":
                src = apply_inverse_raw(raw_filtered, prepared_inv,
                    lambda2=lambda2,
                    method="dSPM", prepared=True,
                    verbose="error").data.astype(np.float32)
            else:
                raise ValueError(f"Unhandled source: {source}")

            if weights.ndim == 1:
                weights = weights[np.newaxis]
            win_samples = int(MULTICOLONY_STEP * SFREQ)
            for wi, row in enumerate(weights):
                top = np.where(row >= np.quantile(row, PERCENTILE))[0]
                t0 = wi * win_samples
                t1 = min(t0 + win_samples, src.shape[1])
                b.append(src[top, t0:t1])

        b = np.concatenate(b)
        
        clf = model_container.get((source, lb)) or \
            NeuralNetClassifier(WordCNN(len(b)), max_epochs=1, lr=0.001, batch_size=1, train_split=None, verbose=0,
                                criterion=nn.CrossEntropyLoss,  # type: ignore[arg-type]
                                device='cuda' if torch.cuda.is_available() else 'cpu')

        clf.criterion__weight = torch.tensor([neg_w, pos_w], dtype=torch.float32)
        clf.partial_fit(b[np.newaxis], np.array([int(binary)], dtype=np.int64))
        
        model_container[(source, lb)] = clf

def train(run, do_colony=True, do_clf=True):
    if do_colony:
        i = 0
        for meg, label_id, run_info in tqdm(run, desc="Training colonies", unit="window"):
            if i == 500:
                break
            
            print(f"Digesting run {i}")

            word = run.id_to_word[int(label_id)]
            label = normalize_word(word)
            raw = create_raw(meg)
            if label in PRIMARY_VOCAB_TO_ID:
                _digest(raw, primary_colonies_words, label)
            if label in MOSES_VOCAB_TO_ID:
                _digest(raw, moses_colonies_words, label)

            i += 1
            if i % 100 == 0:
                print(f"Digested {i} samples...")

    if do_clf:
        i = 0
        for meg, label_id, run_info in tqdm(run, desc="Training CLFs", unit="window"):
            if i == 500:
                break
            
            print(f"Training on run {i}")

            word = run.id_to_word[int(label_id)]
            label = normalize_word(word)
            raw = create_raw(meg)
            if label in PRIMARY_VOCAB_TO_ID:
                _train_clf(raw, primary_colonies_words, primary_band_clfs, label)
            if label in MOSES_VOCAB_TO_ID:
                _train_clf(raw, moses_colonies_words, moses_band_clfs, label)

            i += 1
            if i % 100 == 0:
                print(f"Trained {i} samples...")

def model(meg: np.ndarray):
    raw = create_raw(meg)

    band_data = {}
    for band_name, band in TARGET_BANDS.items():
        low = band["low"]
        high = min(band["high"], SFREQ / 2.0 - 1)
        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4, verbose='error')
        band_data[band_name] = {
            "vol": raw_filtered.get_data().astype(np.float32),
            "inverse": apply_inverse_raw(raw_filtered, prepared_inv,
                lambda2=lambda2, method="dSPM", prepared=True,
                verbose="error").data.astype(np.float32),
        }

    primary_prob = defaultdict(float)
    moses_prob = defaultdict(float)

    for vocab, colonies, clfs, prob in [
        (PRIMARY_VOCAB_TO_ID, primary_colonies_words, primary_band_clfs, primary_prob),
        (MOSES_VOCAB_TO_ID, moses_colonies_words, moses_band_clfs, moses_prob),
    ]:
        for source in ["vol", "inverse"]:
            for word in vocab:
                k = (source, word)
                
                if k not in clfs:
                    continue
                
                channels = []
                
                for band_name in TARGET_BANDS:
                    colony = colonies.get((source, band_name, word))
                    
                    if colony is None:
                        continue
                    
                    show_colony(colony, name=f"{source}_{band_name}_{word}")
                    
                    weights = colony.pos_weights()
                    
                    if weights.ndim == 1:
                        weights = weights[np.newaxis]
                    win_samples = int(MULTICOLONY_STEP * SFREQ)
                    for wi, row in enumerate(weights):
                        top = np.where(row >= np.quantile(row, PERCENTILE))[0]
                        src = band_data[band_name][source]
                        t0 = wi * win_samples
                        t1 = min(t0 + win_samples, src.shape[1])
                        channels.append(src[top, t0:t1])
                
                if not channels:
                    continue
                
                prob[word] += clfs[k].predict_proba(np.concatenate(channels)[np.newaxis])[0, 1]

    primary_list = [0.0] * 50
    moses_list = [0.0] * 50

    for word, id_ in PRIMARY_VOCAB_TO_ID.items():
        primary_list[id_] = primary_prob[word]
    for word, id_ in MOSES_VOCAB_TO_ID.items():
        moses_list[id_] = moses_prob[word]

    return primary_list, moses_list, primary_prob, moses_prob

def validate(run):    
    success = 0
    fail = 0

    n_val = 0
    for meg, label_id, run_info in tqdm(run, desc="Validating", unit="window"):
        word = run.id_to_word[int(label_id)]
        label = normalize_word(word)

        if label not in PRIMARY_VOCAB_TO_ID and label not in MOSES_VOCAB_TO_ID:
            continue

        n_val += 1
        if n_val % 50 == 0:
            total = success + fail
            print(f"Validating {n_val}... {success}/{total} ({success/total*100:.1f}%)" if total else f"Validating {n_val}...")

        _, _, p, m = model(meg)
        print("Primary", p)
        print("Moses", m)
        all_p = nlargest(50, p, key=p.get)
        all_m = nlargest(50, m, key=m.get)
        top_p = nlargest(10, p, key=p.get)
        top_m = nlargest(10, m, key=m.get)
        
        if label in PRIMARY_VOCAB_TO_ID:
            if label in top_p:
                print(f"\tPositive-Primary: '{label}' is in top 10 ({top_p.index(label) + 1}th)")
                success += 1
            else:
                print(f"\tNegative-Primary: '{label}' is NOT in top 10 ({all_p.index(label) + 1}th)")
                fail += 1
        if label in MOSES_VOCAB_TO_ID:
            if label in top_m:
                print(f"\tPositive-Moses: '{label}' is in top 10 ({top_m.index(label) + 1}th)")
                success += 1
            else:
                print(f"\tNegative-Moses: '{label}' is NOT in top 10 ({all_m.index(label) + 1}th)")
                fail += 1
    
    print(f"Validation: {success} successes, {fail} failures ({success / (success + fail) * 100:.2f}% accuracy)")

def main():
    for i, run_key in enumerate(TRAIN_RUNS):
        one_run = LibriBrainWord(
            data_path=str(DATA_PATH),
            include_run_keys=[run_key],
            tmin=TMIN,
            tmax=TMAX,
            standardize=False,     # show raw-ish values for this first look
            include_info=True,     # also return a dict with the word string, onset, etc.
            preload_files=False,   # download lazily instead of all-at-once
        )
        
        train(one_run)
        save_colony_state(f"pnpl/models/colony_run{i}.pt")
        
        print(f"Finished training run {i}, saving and validating...")
        
        for j, run in enumerate(VALIDATION_RUNS):
            one_run = LibriBrainWord(
                data_path=str(DATA_PATH),
                include_run_keys=[run],
                tmin=TMIN,
                tmax=TMAX,
                standardize=False,     # show raw-ish values for this first look
                include_info=True,     # also return a dict with the word string, onset, etc.
                preload_files=False,   # download lazily instead of all-at-once
            )
            
            validate(one_run)

    for i, run in enumerate(TEST_RUNS):
        one_run = LibriBrainWord(
            data_path=str(DATA_PATH),
            include_run_keys=[run],
            tmin=TMIN,
            tmax=TMAX,
            standardize=False,     # show raw-ish values for this first look
            include_info=True,     # also return a dict with the word string, onset, etc.
            preload_files=False,   # download lazily instead of all-at-once
        )
        
        validate(one_run)

if __name__ == "__main__":
    main()

sys.exit()

holdout = LibriBrainCompetitionHoldout(track="deep")   # "deep" or "broad"

primary, moses = [], []
for meg, _meta in holdout.iter_windows(batch_size=None):  # meg: (B, 306, 250)
    p, m, _, _ = model(meg) # need to account for tmax                              # each (B, 50) probabilities
    primary.append(p); moses.append(m)

write_submission(
    "submission.csv",
    indices=holdout.indices,                 # required row order
    primary_probs=np.concatenate(primary),   # (N, 50) competition vocab, scored
    secondary_probs=np.concatenate(moses),   # (N, 50) Moses-50, optional/secondary
)
