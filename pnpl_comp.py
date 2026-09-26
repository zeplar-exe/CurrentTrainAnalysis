from collections import defaultdict
from itertools import groupby, batched
import pickle

from mne.minimum_norm import apply_inverse_raw, prepare_inverse_operator
import numpy as np
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
import sys
from pathlib import Path
import mne
import torch
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
from pnpl_word_durations import word_sd_duration, word_mean_duration
import torch.nn as nn

warnings.filterwarnings(
    action="ignore", 
    category=RuntimeWarning, 
    message=".*is longer than the signal.*"
)

mne.set_log_level('ERROR')

FIF = "~/.cache/huggingface/hub/datasets--pnpl--LibriBrain/snapshots/5a7c332b34fc7be329c4df3e527a41f67bfd878f/Sherlock1/sub-0/ses-1/meg/sub-0_ses-1_task-Sherlock1_run-1_meg.fif"

TMIN, TMAX = 0.0, 1.0
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

PERCENTILE = 0.975

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
        # scale width to the input: vol is ~40 channels, inverse is ~2500
        w1 = int(np.clip(n_channels // 4, 32, 512))
        w2 = max(w1 // 2, 16)
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, w1, kernel_size=5, padding=2),
            nn.BatchNorm1d(w1),
            nn.ReLU(),
            nn.Conv1d(w1, w1, kernel_size=5, padding=2),
            nn.BatchNorm1d(w1),
            nn.ReLU(),
            nn.Conv1d(w1, w2, kernel_size=3, padding=1),
            nn.BatchNorm1d(w2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(w2, 2),
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

def label_duration(label: str):
    return word_mean_duration(label, default=0.45) + 2*word_sd_duration(label, default=0.0)

def pad_or_truncate(x, length, axis=-1, value=0.0):
    n = x.shape[axis]
    if n == length:
        return x
    if n > length:
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(0, length)
        return x[tuple(sl)]
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, length - n)
    return np.pad(x, pad, constant_values=value)

def _digest(raw: mne.io.RawArray, colony_container: dict[tuple[str, str, str], MultiColony], label: str):
    for band_name, band in TARGET_BANDS.items():
        low = band["low"]
        high = min(band["high"], SFREQ / 2.0 - 1)

        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=1, verbose='error')
        raw_filtered.crop(tmin=0.0, tmax=label_duration(label))

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

def _collect_sample(band_data: dict[str, dict[str, np.ndarray]], colony_container: dict[tuple[str, str, str], MultiColony], label: str) -> dict[tuple[str, str], tuple[np.ndarray, int]]:
    label_distribution[label] += 1
    result = {}

    band_grouped = groupby(sorted(colony_container.items(), key=lambda x: (x[0][0], x[0][2])), lambda x: (x[0][0], x[0][2]))
    for (source, lb), group in band_grouped:
        binary = int(lb == label)
        b = []

        for (_, band_name, _), colony in group:
            weights = colony.pos_weights()
            src = band_data[band_name][source]

            if weights.ndim == 1:
                weights = weights[np.newaxis]
            spans = []
            for wi, row in enumerate(weights):
                t0 = round(wi * MULTICOLONY_STEP * SFREQ)
                t1 = min(round((wi + 1) * MULTICOLONY_STEP * SFREQ), src.shape[1])
                if t0 >= t1:
                    break
                top = np.where(row >= np.quantile(row, PERCENTILE))[0]
                spans.append(src[top, t0:t1])
            b.append(np.concatenate(spans, axis=1))

        result[(source, lb)] = (np.concatenate(b), binary)
    return result


def _fit_clfs(buffers: dict[tuple[str, str], list[tuple[np.ndarray, int]]], model_container: dict[tuple[str, str], NeuralNetClassifier], epochs=50, batch_size=32):
    for (source, lb), samples in buffers.items():
        lb_dur_index = int(label_duration(lb) * SFREQ) + 1
        X = np.stack([pad_or_truncate(s[0], lb_dur_index) for s in samples])
        y = np.array([s[1] for s in samples], dtype=np.int64)

        pos_count = y.sum()
        neg_count = len(y) - pos_count
        pos_w = neg_count / len(y) if len(y) > 0 else 0.5
        neg_w = pos_count / len(y) if len(y) > 0 else 0.5

        n_channels = X.shape[1]

        clf = model_container.get((source, lb)) or \
            NeuralNetClassifier(WordCNN(n_channels), max_epochs=epochs, lr=0.001, batch_size=batch_size, train_split=None, verbose=0,
                                criterion=nn.CrossEntropyLoss,  # type: ignore[arg-type]
                                device='cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

        if pos_count == 0 or neg_count == 0:
            clf.criterion__weight = None # unweighted: negatives count fully
        else:
            clf.criterion__weight = torch.tensor([neg_w, pos_w], dtype=torch.float32)

        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
        if hasattr(clf, 'module_'):
            clf.module_.to(device)
            for st in clf.optimizer_.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device)
        clf.partial_fit(X, y)
        clf.module_.to('cpu')
        for st in clf.optimizer_.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_container[(source, lb)] = clf
        last_loss = clf.history[-1]['train_loss'] if clf.history else float('nan')
        print(f"Fit {source}/{lb}: {len(y)} samples, {pos_count} pos, {neg_count} neg, loss={last_loss:.4f}")

def train(run, do_colony=True, do_clf=True):
    if do_colony:
        i = 0
        for meg, label in tqdm(run, desc="Feeding colonies", unit="window"):
            raw = create_raw(meg)
            if label in PRIMARY_VOCAB_TO_ID:
                _digest(raw, primary_colonies_words, label)
            if label in MOSES_VOCAB_TO_ID:
                _digest(raw, moses_colonies_words, label)

            i += 1
            if i % 100 == 0:
                print(f"Digested {i} samples...")
        
        for (source, band, label), colony in primary_colonies_words.items():
            if source == "inverse":
                show_colony(colony, name=f"{source}_{band}_{label}", output=f"./pnpl/colonies/primary-{source}_{band}_{label}.html")
        for (source, band, label), colony in primary_colonies_words.items():
            if source == "inverse":
                show_colony(colony, name=f"{source}_{band}_{label}", output=f"./pnpl/colonies/moses-{source}_{band}_{label}.html")
    if do_clf:
        for batch in tqdm(batched(run, 100), desc="Collecting samples", unit="window"):
            primary_buffers: dict[tuple[str, str], list[tuple[np.ndarray, int]]] = defaultdict(list)
            moses_buffers: dict[tuple[str, str], list[tuple[np.ndarray, int]]] = defaultdict(list)
            
            for meg, label in batch:
                raw = create_raw(meg)

                band_data = {}
                for band_name, band in TARGET_BANDS.items():
                    low = band["low"]
                    high = min(band["high"], SFREQ / 2.0 - 1)
                    raw_filtered = raw.copy()
                    raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=1, verbose='error')
                    raw_filtered.crop(tmin=0.0, tmax=label_duration(label))
                    band_data[band_name] = {
                        "vol": raw_filtered.get_data().astype(np.float32), 
                        "inverse": apply_inverse_raw(raw_filtered, prepared_inv,
                            lambda2=lambda2, method="dSPM", prepared=True,
                            verbose="error").data.astype(np.float32),
                    }

                if label in PRIMARY_VOCAB_TO_ID:
                    for k, v in _collect_sample(band_data, primary_colonies_words, label).items():
                        primary_buffers[k].append(v)
                if label in MOSES_VOCAB_TO_ID:
                    for k, v in _collect_sample(band_data, moses_colonies_words, label).items():
                        moses_buffers[k].append(v)

            _fit_clfs(primary_buffers, primary_band_clfs)
            _fit_clfs(moses_buffers, moses_band_clfs)

def model(meg: np.ndarray):
    raw = create_raw(meg)

    band_data = {}
    for band_name, band in TARGET_BANDS.items():
        low = band["low"]
        high = min(band["high"], SFREQ / 2.0 - 1)
        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=1, verbose='error')
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
                    
                    weights = colony.pos_weights()
                    
                    if weights.ndim == 1:
                        weights = weights[np.newaxis]
                    src = band_data[band_name][source]
                    spans = []
                    for wi, row in enumerate(weights):
                        t0 = round(wi * MULTICOLONY_STEP * SFREQ)
                        # according to Claude, MNE's crop is int(T * SFREQ) + 1, to do with include_max, tbd
                        t1 = min(int((wi + 1) * MULTICOLONY_STEP * SFREQ), src.shape[1], int(label_duration(word) * SFREQ) + 1)
                        if t0 >= t1:
                            break
                        top = np.where(row >= np.quantile(row, PERCENTILE))[0]
                        spans.append(src[top, t0:t1])
                    lb_dur_index = int(label_duration(word) * SFREQ) + 1
                    channels.append(pad_or_truncate(np.concatenate(spans, axis=1), lb_dur_index))
                
                if not channels:
                    continue
                
                clf = clfs[k]
                clf.module_.to(clf.device)
                prob[word] += clf.predict_proba(np.concatenate(channels)[np.newaxis])[0, 1]
                clf.module_.to('cpu')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
    for meg, label in tqdm(run, desc="Validating", unit="window"):
        n_val += 1
        if n_val % 50 == 0:
            total = success + fail
            print(f"Validating {n_val}... {success}/{total} ({success/total*100:.1f}%)" if total else f"Validating {n_val}...")

        _, _, p, m = model(meg)
        all_p = nlargest(50, p, key=p.get)
        all_m = nlargest(50, m, key=m.get)
        top_p = nlargest(10, p, key=p.get)
        top_m = nlargest(10, m, key=m.get)
        
        if label in PRIMARY_VOCAB_TO_ID:
            if label in top_p:
                print(f"\tPositive-Primary: '{label}' IS in top 10 ({top_p.index(label) + 1}th)")
                success += 1
            else:
                print(f"\tNegative-Primary: '{label}' is NOT in top 10 ({all_p.index(label) + 1}th)")
                fail += 1
        if label in MOSES_VOCAB_TO_ID:
            if label in top_m:
                print(f"\tPositive-Moses: '{label}' IS in top 10 ({top_m.index(label) + 1}th)")
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
        
        one_run = [(r[0], normalize_word(one_run.id_to_word[int(r[1])])) for r in one_run]
        one_run = [r for r in one_run if r[1] in PRIMARY_VOCAB_TO_ID or r[1] in MOSES_VOCAB_TO_ID]
        
        train(one_run)
        print(f"Finished training run {i}, saving and validating...")
        save_colony_state(f"pnpl/models/colony_run{i}.pt")
        #load_colony_state("pnpl/models/colony_run0.pt")
        print("\tCurrent Label Distribution:", dict(label_distribution))
        
        
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
            
            one_run = [(r[0], normalize_word(one_run.id_to_word[int(r[1])])) for r in one_run]
            one_run = [r for r in one_run if r[1] in PRIMARY_VOCAB_TO_ID or r[1] in MOSES_VOCAB_TO_ID]
            
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
        
        one_run = [(r[0], normalize_word(one_run.id_to_word[int(r[1])])) for r in one_run]
        one_run = [r for r in one_run if r[1] in PRIMARY_VOCAB_TO_ID or r[1] in MOSES_VOCAB_TO_ID]
        
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
