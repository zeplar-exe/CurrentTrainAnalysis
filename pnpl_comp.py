import argparse
import json
from collections import defaultdict
from itertools import groupby
import pickle
import re

from mne.minimum_norm import prepare_inverse_operator
from mne.minimum_norm.inverse import _assemble_kernel
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
import torch.nn.functional as F

warnings.filterwarnings(
    action="ignore", 
    category=RuntimeWarning, 
    message=".*is longer than the signal.*"
)

mne.set_log_level('ERROR')

FIF = "~/.cache/huggingface/hub/datasets--pnpl--LibriBrain/snapshots/5a7c332b34fc7be329c4df3e527a41f67bfd878f/Sherlock1/sub-0/ses-1/meg/sub-0_ses-1_task-Sherlock1_run-1_meg.fif"

TMIN, TMAX = 0.1, 2.0
DATA_PATH = Path(".") / "pnpl" / "libribrain_word"
TRAIN_RUNS = [("0", str(s), "Sherlock1", "1") for s in range(1, 7 + 1)]   # sessions 1-7
VALIDATION_RUNS = [("0", str(s), "Sherlock1", "1") for s in range(8, 9 + 1)]   # sessions 8-9
TEST_RUNS  = [("0", str(s), "Sherlock1", "1") for s in range(10, 10 + 1)]  # session 10

PRIMARY_VOCAB = load_vocabulary("primary")      # the 50 competition words, in order
MOSES_VOCAB = load_vocabulary("moses")        # the 50 Moses words (secondary metric)

def batched(seq, n):
    # itertools.batched is 3.12+
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

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

# classifier inputs are (band, stage, vertex, sample): one block per multicolony stage
STAGE_LEN = int(MULTICOLONY_STEP * SFREQ)
CLASSIFIER = "stage"

# vol only, no inverse: every stage gets all 306 channels, each scaled by its colony pos_weight
# instead of the top-PERCENTILE cutoff. mags are first brought to grad scale (median |x| ratio
# ~21 in every band) so the colony weight, not the sensor type, sets each channel's size
VOL_WEIGHTED = True
MAG_TO_GRAD = 21.0

# classifiers are fit after all colonies are built, from band-filtered sensor windows cached
# to disk (cropped to their label's duration)
CACHE_DIR = Path("./pnpl/cache")
CACHE_SCALE = 1e11  # float16 cache: puts mags (~2e-13 T) and grads (~5e-12 T/m) well inside float16 range

MODEL_STATE_PATH = Path("./pnpl/model_state.pkl")

info_fif = mne.io.read_info(FIF)
info_fif = mne.pick_info(info_fif, mne.pick_types(info_fif, meg=True, exclude=[]))
with info_fif._unlock():
    info_fif['sfreq'] = SFREQ

VOL_SCALE = np.ones(len(info_fif.ch_names), dtype=np.float32)
VOL_SCALE[mne.pick_types(info_fif, meg="mag", exclude=[])] = MAG_TO_GRAD

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

# dSPM for any vertex subset: that subset's kernel rows applied to the sensors, the three
# orientations pooled, then noise-normalized -- identical to apply_inverse_raw's output rows
_kernel, _noise_norm, _, _ = _assemble_kernel(prepared_inv, None, "dSPM", None)
INV_KERNEL = _kernel.reshape(-1, 3, _kernel.shape[1]).astype(np.float32)  # (n_src, 3, n_chan)
INV_NOISE = np.asarray(_noise_norm).ravel().astype(np.float32)            # (n_src,)

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
        n_spatial = min(n_channels, 128)
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, n_spatial, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_spatial),
            nn.Conv1d(n_spatial, w1, kernel_size=5, padding=2),
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


class StageCNN(nn.Module):
    """Continuous-input classifier shaped around the multicolony stages.

    Each (band, stage) block of colony-selected vertices is encoded on its own:
    a spatial projection per (band, stage), per-band temporal filters that never
    cross a stage edge, then per filter: log power inside the stage plus the signed
    filter output averaged into a few time bins. The stage embeddings stay in onset
    order and the head is factored: a weight per stage times a linear map over
    (band, filter, feature), so timing relative to onset is kept without a free
    weight for every (stage, feature) pair -- ns + 2*n_feat weights instead of
    2*ns*n_feat, against tens of positives per word per session.

    > Thanks Claude
    """

    def __init__(self, n_bands, n_stages, n_vertices, n_components=8, n_filters=8, kernel_size=5, n_bins=3):
        super().__init__()
        k = min(n_vertices, n_components)
        self.k = k
        self.n_bins = n_bins
        # grouped by (band, stage): each stage selects its own vertices (in index order),
        # so channel i is a different vertex per stage and spatial weights can't be shared.
        # bands are never summed together either
        self.spatial = nn.Conv1d(n_bands * n_stages * n_vertices, n_bands * n_stages * k,
                                 kernel_size=1, groups=n_bands * n_stages, bias=False)
        # no padding: each stage is its own batch item, so filters only see that stage
        self.temporal = nn.Conv1d(n_bands * k, n_bands * n_filters, kernel_size=kernel_size, groups=n_bands, bias=False)
        n_feat = n_bands * n_filters * (1 + n_bins)
        # log power is scale-free up to an offset (vol is tesla-scale, inverse is dSPM),
        # so standardize each feature against the batch -- i.e. against mostly-negative windows
        self.norm = nn.BatchNorm1d(n_stages * n_feat, affine=False)
        self.stage_w = nn.Parameter(torch.full((n_stages,), n_stages ** -0.5))
        self.head = nn.Linear(n_feat, 2)

    def forward(self, X):
        # X: (batch, n_bands, n_stages, n_vertices, stage_len); all-zero blocks are missing stages
        B, nb, ns, V, S = X.shape
        present = X.abs().amax(dim=(3, 4)) > 0                     # (B, nb, ns)

        x = self.spatial(X.reshape(B, nb * ns * V, S))             # (B, nb*ns*k, S), ordered (band, stage, k)
        x = x.reshape(B, nb, ns, self.k, S).permute(0, 2, 1, 3, 4)
        x = self.temporal(x.reshape(B * ns, nb * self.k, S))       # stages -> batch; (B*ns, nb*F, S - kernel + 1)
        power = x.pow(2).mean(-1).clamp_min(1e-30)                 # (B*ns, nb*F)
        # signed evoked shape: binned means over the stage RMS, so it's scale-free like log power
        # (raw vol means are ~1e-13 and would vanish under BatchNorm's eps)
        shape = F.adaptive_avg_pool1d(x, self.n_bins) / power.sqrt()[..., None]
        x = torch.cat([torch.log(power)[..., None], shape], dim=-1)  # (B*ns, nb*F, 1 + n_bins)
        x = x.reshape(B, ns, nb, -1).permute(0, 2, 1, 3)           # (B, nb, ns, F * (1 + n_bins))

        # keep missing stages out of the batch statistics, then out of the logit
        p = present[..., None].expand_as(x)
        fill = (x * p).sum(0) / p.sum(0).clamp_min(1)
        x = torch.where(p, x, fill.detach())
        x = self.norm(x.reshape(B, -1)).reshape(B, nb, ns, -1) * p
        x = x.permute(0, 2, 1, 3).reshape(B, ns, -1)               # (B, ns, nb * F * (1 + n_bins))
        return self.head(torch.einsum("bsf,s->bf", x, self.stage_w))

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
    return min(word_mean_duration(label, default=0.45) + 2*word_sd_duration(label, default=0.0) + 0.3, TMAX - TMIN)

def n_stages_for(word: str):
    return max(1, int(label_duration(word) / MULTICOLONY_STEP))

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

def _to_clf_input(X: np.ndarray):
    if CLASSIFIER == "cnn":
        N, nb, ns, V, S = X.shape
        return np.ascontiguousarray(X.transpose(0, 1, 3, 2, 4).reshape(N, nb * V, ns * S))
    return X

def _source_rows(sensors: np.ndarray, source: str, weights: np.ndarray, t0: int, t1: int):
    """One stage of the colony's channels (weights: that stage's pos_weights row): sensor rows for vol,
    dSPM of just those vertices for inverse. With VOL_WEIGHTED, vol is every channel scaled by its weight."""
    x = sensors[:, t0:t1]
    if source == "vol" and VOL_WEIGHTED:
        return x * (VOL_SCALE * weights)[:, None]
    rows = np.where(weights >= np.quantile(weights, PERCENTILE))[0]
    if source == "vol":
        return x[rows]
    sol = INV_KERNEL[rows] @ x                                   # (V, 3, S)
    return np.sqrt((sol ** 2).sum(1)) * INV_NOISE[rows, None]

def _digest(raw: mne.io.RawArray, colony_container: dict[tuple[str, str, str], MultiColony], label: str):
    band_data = {}
    for band_name, band in TARGET_BANDS.items():
        low = band["low"]
        high = min(band["high"], SFREQ / 2.0 - 1)

        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=1, verbose='error')
        raw_filtered.crop(tmin=0.0, tmax=min(label_duration(label), raw_filtered.times[-1]))
        band_data[band_name] = raw_filtered.get_data()

        new_colonies = compute_gain(prepared_inv, raw_filtered,
            lambda2, TIMESTEP, MULTICOLONY_STEP, None,
            include_vol=True, include_csd=False, include_inverse=not VOL_WEIGHTED,
            include_pos=True, include_neg=False, use_epochs=False)

        for (source, _), new_colony in new_colonies.items():
            k = (source, band_name, label)
            if k in colony_container:
                colony_container[k].merge(new_colony)
            else:
                colony_container[k] = new_colony
    return band_data

def _collect_sample(band_data: dict[str, np.ndarray], colony_container: dict[tuple[str, str, str], MultiColony], label: str) -> dict[tuple[str, str], tuple[np.ndarray, int]]:
    result = {}

    band_grouped = groupby(sorted(colony_container.items(), key=lambda x: (x[0][0], x[0][2])), lambda x: (x[0][0], x[0][2]))
    for (source, lb), group in band_grouped:
        binary = int(lb == label)
        b = []

        for (_, band_name, _), colony in group:
            weights = colony.pos_weights()
            src = band_data[band_name]

            if weights.ndim == 1:
                weights = weights[np.newaxis]
            spans = []
            for wi, row in enumerate(weights):
                t0 = round(wi * MULTICOLONY_STEP * SFREQ)
                t1 = t0 + STAGE_LEN
                if t1 > src.shape[1]:
                    break
                spans.append(_source_rows(src, source, row, t0, t1))
            # stages on their own axis; pad (zeros, masked in StageCNN) or cut to lb's stage count
            b.append(pad_or_truncate(np.stack(spans), n_stages_for(lb), axis=0))

        result[(source, lb)] = (np.stack(b), binary)
    return result


def _fit_clfs(buffers: dict[tuple[str, str], list[tuple[np.ndarray, int]]], model_container: dict[tuple[str, str], NeuralNetClassifier], epochs=50, batch_size=32):
    for (source, lb), samples in buffers.items():
        X = _to_clf_input(np.stack([s[0] for s in samples]))
        y = np.array([s[1] for s in samples], dtype=np.int64)

        if CLASSIFIER == "stage" and len(y) < batch_size:
            print(f"Skipping fit {source}/{lb}: {len(y)} samples < batch size {batch_size}")
            continue

        pos_count = y.sum()
        neg_count = len(y) - pos_count
        pos_w = neg_count / len(y) if len(y) > 0 else 0.5
        neg_w = pos_count / len(y) if len(y) > 0 else 0.5

        clf = model_container.get((source, lb))
        if clf is None:
            if CLASSIFIER == "stage":
                module = StageCNN(n_bands=X.shape[1], n_stages=X.shape[2], n_vertices=X.shape[3])
                # its BatchNorm is over a flat feature vector, which can't train on a batch of 1
                extra = {"iterator_train__drop_last": True}
            else:
                module = WordCNN(X.shape[1])
                extra = {}
            clf = NeuralNetClassifier(module, max_epochs=epochs, lr=0.001, batch_size=batch_size, train_split=None, verbose=0,
                                      optimizer=torch.optim.AdamW,
                                      criterion=nn.CrossEntropyLoss,  # type: ignore[arg-type]
                                      device='cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu',
                                      **extra)

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

def _cache_path(i: int) -> Path:
    return CACHE_DIR / f"train_run{i}"

def train(run, cache_path: Path):
    """Feed the colonies, caching each window's band-filtered sensor data (cropped to its label's duration)."""
    blocks, lengths, labels = [], [], []
    i = 0
    for meg, label in tqdm(run, desc="Feeding colonies", unit="window"):
        raw = create_raw(meg)
        if label in PRIMARY_VOCAB_TO_ID:
            band_data = _digest(raw, primary_colonies_words, label)
        if label in MOSES_VOCAB_TO_ID:
            band_data = _digest(raw, moses_colonies_words, label)
        label_distribution[label] += 1

        x = np.stack(list(band_data.values()))                              # (bands, chan, T)
        blocks.append((x.transpose(2, 0, 1) * CACHE_SCALE).astype(np.float16))
        lengths.append(x.shape[-1])
        labels.append(label)

        i += 1
        if i % 100 == 0:
            print(f"Digested {i} samples...")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path.with_suffix(".npy"), np.concatenate(blocks))          # (sum T, bands, chan)
    cache_path.with_suffix(".json").write_text(json.dumps({"lengths": lengths, "labels": labels}))

    for (source, band, label), colony in primary_colonies_words.items():
        if source == "inverse":
            show_colony(colony, name=f"{source}_{band}_{label}", output=f"./pnpl/colonies/primary-{source}_{band}_{label}.html")
    for (source, band, label), colony in primary_colonies_words.items():
        if source == "inverse":
            show_colony(colony, name=f"{source}_{band}_{label}", output=f"./pnpl/colonies/moses-{source}_{band}_{label}.html")

def fit_from_cache(cache_paths: list[Path], epochs=15):
    """The classifier loop train() used to run per run, now over every cached run with the final colonies.

    Each epoch is one pass over all cached windows in shuffled 100-window chunks, one partial_fit epoch
    per chunk, so no chunk is trained to convergence before the next one arrives."""
    cached = []
    for path in cache_paths:
        data = np.load(path.with_suffix(".npy"), mmap_mode="r")
        meta = json.loads(path.with_suffix(".json").read_text())
        offsets = np.cumsum([0] + meta["lengths"])
        cached += [(data[o:o + n], label) for o, n, label in zip(offsets, meta["lengths"], meta["labels"])]

    for epoch in range(epochs):
        order = [cached[i] for i in np.random.permutation(len(cached))]
        for batch in tqdm(batched(order, 100), desc=f"Epoch {epoch + 1}/{epochs}", unit="chunk"):
            primary_buffers: dict[tuple[str, str], list[tuple[np.ndarray, int]]] = defaultdict(list)
            moses_buffers: dict[tuple[str, str], list[tuple[np.ndarray, int]]] = defaultdict(list)

            for x, label in batch:
                x = np.asarray(x, dtype=np.float32).transpose(1, 2, 0) / CACHE_SCALE   # (bands, chan, T)
                band_data = dict(zip(TARGET_BANDS, x))

                if label in PRIMARY_VOCAB_TO_ID:
                    for k, v in _collect_sample(band_data, primary_colonies_words, label).items():
                        primary_buffers[k].append(v)
                if label in MOSES_VOCAB_TO_ID:
                    for k, v in _collect_sample(band_data, moses_colonies_words, label).items():
                        moses_buffers[k].append(v)

            _fit_clfs(primary_buffers, primary_band_clfs, epochs=1)
            _fit_clfs(moses_buffers, moses_band_clfs, epochs=1)

def model(meg: np.ndarray):
    raw = create_raw(meg)

    band_data = {}
    for band_name, band in TARGET_BANDS.items():
        low = band["low"]
        high = min(band["high"], SFREQ / 2.0 - 1)
        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=1, verbose='error')
        band_data[band_name] = raw_filtered.get_data().astype(np.float32)

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
                    src = band_data[band_name]
                    spans = []
                    for wi, row in enumerate(weights):
                        t0 = round(wi * MULTICOLONY_STEP * SFREQ)
                        t1 = t0 + STAGE_LEN
                        if t1 > src.shape[1]:
                            break
                        spans.append(_source_rows(src, source, row, t0, t1))
                    channels.append(pad_or_truncate(np.stack(spans), n_stages_for(word), axis=0))

                if not channels:
                    continue

                clf = clfs[k]
                clf.module_.to(clf.device)
                prob[word] += clf.predict_proba(_to_clf_input(np.stack(channels)[np.newaxis]))[0, 1]
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

def validate(run, scores_path=None):
    success = 0
    fail = 0

    out = None
    if scores_path is not None:
        scores_path = Path(scores_path)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        out = open(scores_path, "w")
        out.write(json.dumps({
            "primary_vocab": [PRIMARY_ID_TO_VOCAB[i] for i in range(len(PRIMARY_ID_TO_VOCAB))],
            "moses_vocab": [MOSES_ID_TO_VOCAB[i] for i in range(len(MOSES_ID_TO_VOCAB))],
            "train_counts": dict(label_distribution),
        }) + "\n")
        out.flush()

    n_val = 0
    for meg, label in tqdm(run, desc="Validating", unit="window"):
        n_val += 1
        if n_val % 50 == 0:
            total = success + fail
            print(f"Validating {n_val}... {success}/{total} ({success/total*100:.1f}%)" if total else f"Validating {n_val}...")

        primary_list, moses_list, p, m = model(meg)
        if out is not None:
            out.write(json.dumps({
                "label": label,
                "primary": [round(float(s), 6) for s in primary_list],
                "moses": [round(float(s), 6) for s in moses_list],
            }) + "\n")
            out.flush()
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

    if out is not None:
        out.close()
        print(f"Saved score spectra to {scores_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, default=None,
                        help="saved state to resume from, e.g. pnpl/models/colony_run2.pt")
    args = parser.parse_args()

    start = 0
    if args.resume is not None:
        load_colony_state(args.resume)
        m = re.search(r"colony_run(\d+)", args.resume.stem)
        start = int(m.group(1)) + 1 if m else 0
        print(f"Resumed from {args.resume}, starting at training run {start}")

    for i, run_key in enumerate(TRAIN_RUNS):
        if i < start:
            continue
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
        
        train(one_run, _cache_path(i))
        print(f"Finished colonies for training run {i}, saving...")
        save_colony_state(f"pnpl/models/colony_run{i}.pt")

    # classifiers only once the colonies are final, so every fit sees the same vertex selection
    fit_from_cache([_cache_path(i) for i in range(len(TRAIN_RUNS))])
    save_colony_state("pnpl/models/colony_final.pt")
    print("\tLabel Distribution:", dict(label_distribution))

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

        validate(one_run, f"pnpl/scores/final_val_ses{run[1]}.jsonl")

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
        
        validate(one_run, f"pnpl/scores/test_ses{run[1]}.jsonl")

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
