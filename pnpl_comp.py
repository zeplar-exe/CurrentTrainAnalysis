from collections import defaultdict
import pickle

from mne.minimum_norm import prepare_inverse_operator
import numpy as np
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
import sys
from pathlib import Path
from mne.datasets import sample
import mne
import os
from heapq import nlargest
from pnpl.datasets import LibriBrainWord
from pnpl.competition import load_vocabulary
from sklearn.linear_model import SGDClassifier
from colony import Colony, MultiColony, compute_gain, setup_inverse
from colony_viewer import show_colony
from core import BANDS
from tqdm import tqdm
import mne
from pathlib import Path
import warnings

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

MODEL_STATE_PATH = Path("./pnpl/model_state.pkl")

info_fif = mne.io.read_info(FIF)
info_fif = mne.pick_info(info_fif, mne.pick_types(info_fif, meg=True, exclude=[]))
with info_fif._unlock():
    info_fif['sfreq'] = SFREQ

def create_raw(meg):
    raw = mne.io.RawArray(meg, info_fif, verbose=False)
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

primary_band_pos_weights = {}
moses_band_pos_weights = {}
primary_band_neg_weights = {}
moses_band_neg_weights = {}
primary_band_pos_clfs = {}
moses_band_pos_clfs = {}
primary_band_neg_clfs = {}
moses_band_neg_clfs = {}
# this gets overwritten

def digest(raw, colony_container: dict[tuple[str, str, str], MultiColony], label):
    for band_name, band in TARGET_BANDS.items():
        low = band["low"]
        high = min(band["high"], SFREQ / 2.0 - 1)

        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4, verbose=False)

        new_colonies = compute_gain(prepared_inv, raw_filtered,
            lambda2, TIMESTEP, MULTICOLONY_STEP, None, 
            include_vol=True, include_csd=False, include_inverse=True, 
            include_pos=True, include_neg=True, use_epochs=False)

        for (source, _), new_colony in new_colonies.items():
            k = (source, band_name, label)
            if k in colony_container:
                colony_container[k].merge(new_colony)
            else:
                colony_container[k] = new_colony

def save_model(f: str | Path = MODEL_STATE_PATH):
    path = Path(f)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "primary_colonies_words": primary_colonies_words,
        "moses_colonies_words": moses_colonies_words,
        "primary_band_pos_weights": primary_band_pos_weights,
        "moses_band_pos_weights": moses_band_pos_weights,
        "primary_band_neg_weights": primary_band_neg_weights,
        "moses_band_neg_weights": moses_band_neg_weights,
        "primary_band_pos_clfs": primary_band_pos_clfs,
        "moses_band_pos_clfs": moses_band_pos_clfs,
        "primary_band_neg_clfs": primary_band_neg_clfs,
        "moses_band_neg_clfs": moses_band_neg_clfs,
        "metadata": {
            "sfreq": SFREQ,
            "timestep": TIMESTEP,
            "multicolony_step": MULTICOLONY_STEP,
            "target_bands": list(TARGET_BANDS.keys()),
        },
    }

    with path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(f: str | Path = MODEL_STATE_PATH):
    path = Path(f)
    with path.open("rb") as handle:
        state = pickle.load(handle)

    global primary_colonies_words, moses_colonies_words
    global primary_band_pos_weights, moses_band_pos_weights
    global primary_band_neg_weights, moses_band_neg_weights
    global primary_band_pos_clfs, moses_band_pos_clfs
    global primary_band_neg_clfs, moses_band_neg_clfs

    primary_colonies_words = state["primary_colonies_words"]
    moses_colonies_words = state["moses_colonies_words"]
    primary_band_pos_weights = state["primary_band_pos_weights"]
    moses_band_pos_weights = state["moses_band_pos_weights"]
    primary_band_neg_weights = state["primary_band_neg_weights"]
    moses_band_neg_weights = state["moses_band_neg_weights"]
    primary_band_pos_clfs = state["primary_band_pos_clfs"]
    moses_band_pos_clfs = state["moses_band_pos_clfs"]
    primary_band_neg_clfs = state["primary_band_neg_clfs"]
    moses_band_neg_clfs = state["moses_band_neg_clfs"]

    return state

def train(run):
    i = 0
    for meg, label_id, run_info in tqdm(run, desc="Training", unit="window"):
        if i == 10000:
            break
        
        word = run.id_to_word[int(label_id)]
        label = normalize_word(word)
        raw = create_raw(meg)
        if label in PRIMARY_VOCAB_TO_ID:
            digest(raw, primary_colonies_words, label)
        if label in MOSES_VOCAB_TO_ID:
            digest(raw, moses_colonies_words, label)
            
        i += 1
        
    X_pos = defaultdict(list)
    X_neg = defaultdict(list)
    y = defaultdict(list)
    
    for (source, band, label), colony in primary_colonies_words.items():
        X_pos[(source, band)].append(colony.pos_weights().reshape(-1))
        X_neg[(source, band)].append(colony.neg_weights().reshape(-1))
        y[(source, band)].append(label)
    
    for src, dest, clf_dict in [
        (X_pos, primary_band_pos_weights, primary_band_pos_clfs),
        (X_neg, primary_band_neg_weights, primary_band_neg_clfs),
    ]:
        for (source, band), X in src.items():
            clf = clf_dict.get((source, band)) or SGDClassifier(loss="log_loss", random_state=42)
            labels = np.asarray(y[(source, band)])
            clf.fit(np.asarray(X), labels, classes=np.array(list(PRIMARY_VOCAB_TO_ID.keys())))
            clf_dict[(source, band)] = clf
            weights = np.abs(clf.coef_)
            score_values = np.linalg.norm(weights, axis=1)
            score_values = np.exp(score_values - np.max(score_values)) / np.sum(np.exp(score_values - np.max(score_values)))

            for event_name, score in zip(clf.classes_, score_values):
                dest[(source, band, event_name)] = float(score)
                print("Primary Weights for", source, band, event_name, dest[(source, band, event_name)])

    X_pos = defaultdict(list)
    X_neg = defaultdict(list)
    y = defaultdict(list)
    
    for (source, band, label), colony in moses_colonies_words.items():
        X_pos[(source, band)].append(colony.pos_weights().reshape(-1))
        X_neg[(source, band)].append(colony.neg_weights().reshape(-1))
        y[(source, band)].append(label)
    
    for src, dest, clf_dict in [
        (X_pos, moses_band_pos_weights, moses_band_pos_clfs),
        (X_neg, moses_band_neg_weights, moses_band_neg_clfs),
    ]:
        for (source, band), X in src.items():
            clf = clf_dict.get((source, band)) or SGDClassifier(loss="log_loss", random_state=42)
            labels = np.asarray(y[(source, band)])
            clf.fit(np.asarray(X), labels, classes=np.array(list(MOSES_VOCAB_TO_ID.keys())))
            clf_dict[(source, band)] = clf
            weights = np.abs(clf.coef_)
            score_values = np.linalg.norm(weights, axis=1)
            score_values = np.exp(score_values - np.max(score_values)) / np.sum(np.exp(score_values - np.max(score_values)))

            for event_name, score in zip(clf.classes_, score_values):
                dest[(source, band, event_name)] = float(score)
                print("Moses Weights for", source, band, event_name, dest[(source, band, event_name)])

def compare(ref_multi: MultiColony, pred_multi: MultiColony, weight_key, is_primary) -> float:
    def compare_inner(ref_colony : Colony, pred_colony: Colony) -> tuple[float, float, float, float]:
        ref_pos_weights = ref_colony.pos_weights()
        ref_neg_weights = ref_colony.neg_weights()
        pred_pos_weights = pred_colony.pos_weights()
        pred_neg_weights = pred_colony.neg_weights()
        
        PERCENTILE = 0.75

        pos_lo, pos_hi = np.quantile(pred_pos_weights, PERCENTILE), np.quantile(pred_pos_weights, 0.99)
        neg_lo, neg_hi = np.quantile(pred_neg_weights, 1 - 0.99), np.quantile(pred_neg_weights, 1 - 0.75)
        pos_top = set(np.where((pred_pos_weights >= pos_lo) & (pred_pos_weights <= pos_hi))[0])
        neg_top = set(np.where((pred_neg_weights >= neg_lo) & (pred_neg_weights <= neg_hi))[0])

        ref_pos_lo, ref_pos_hi = np.quantile(ref_pos_weights, PERCENTILE), np.quantile(ref_pos_weights, 0.99)
        ref_neg_lo, ref_neg_hi = np.quantile(ref_neg_weights, 1 - 0.99), np.quantile(ref_neg_weights, 1 - 0.75)
        ref_pos_top = set(np.where((ref_pos_weights >= ref_pos_lo) & (ref_pos_weights <= ref_pos_hi))[0])
        ref_neg_top = set(np.where((ref_neg_weights >= ref_neg_lo) & (ref_neg_weights <= ref_neg_hi))[0])

        pos_union = pos_top | ref_pos_top
        neg_union = neg_top | ref_neg_top
        pos_overlap = len(pos_top & ref_pos_top) / len(pos_union) if pos_union else 0.0
        neg_overlap = len(neg_top & ref_neg_top) / len(neg_union) if neg_union else 0.0

        pos_sel = list(pos_top | ref_pos_top)
        neg_sel = list(neg_top | ref_neg_top)
        pos_distance = np.sqrt(np.sum((pred_pos_weights[pos_sel] - ref_pos_weights[pos_sel]) ** 2)) if pos_sel else 0.0
        neg_distance = np.sqrt(np.sum((pred_neg_weights[neg_sel] - ref_neg_weights[neg_sel]) ** 2)) if neg_sel else 0.0
        
        if is_primary:
            pos_overlap *= primary_band_pos_weights.get(weight_key, 1.0)
            neg_overlap *= primary_band_neg_weights.get(weight_key, 1.0)
            pos_distance *= primary_band_pos_weights.get(weight_key, 1.0)
            neg_distance *= primary_band_neg_weights.get(weight_key, 1.0)
        else:
            pos_overlap *= moses_band_pos_weights.get(weight_key, 1.0)
            neg_overlap *= moses_band_neg_weights.get(weight_key, 1.0)
            pos_distance *= moses_band_pos_weights.get(weight_key, 1.0)
            neg_distance *= moses_band_neg_weights.get(weight_key, 1.0)
        
        return pos_overlap, pos_distance, neg_overlap, neg_distance

    p = 0
    
    for k, ref_colony in enumerate(ref_multi.colonies):
        if k < len(pred_multi.colonies):
            pos_overlap, _, neg_overlap, _ = compare_inner(ref_colony, pred_multi.colonies[k])
            p += pos_overlap + neg_overlap

    return p

def model(meg: np.ndarray, scorer):
    raw = create_raw(meg)
    
    primary_prob = defaultdict(float)
    moses_prob = defaultdict(float)
    
    if scorer == "overlap":
        for band_name, band in TARGET_BANDS.items():
            low = band["low"]
            high = min(band["high"], SFREQ / 2.0 - 1)
            
            raw_filtered = raw.copy()
            raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4, verbose='error')
            
            pred_colonies = compute_gain(prepared_inv, raw_filtered,
                lambda2, TIMESTEP, MULTICOLONY_STEP, None, 
                include_vol=True, include_csd=False, include_inverse=True, 
                include_pos=True, include_neg=True, use_epochs=False)
            
            for (word_source, colony_source, output) in [(PRIMARY_VOCAB_TO_ID, primary_colonies_words, primary_prob), (MOSES_VOCAB_TO_ID, moses_colonies_words, moses_prob)]:
                for word in word_source.keys():
                    acc = 0
                    
                    for source in ["inverse"]:
                        ks = (source, band_name, word)
                        kp = (source, "")
                        
                        if ks not in colony_source:
                            continue
                        
                        #show_colony(colony_source[ks], name=f"Ref: {word} ({source}, {band})")
                        #show_colony(pred_colonies[kp], name=f"Pred: {word} ({source}, {band})")
                        
                        acc += compare(colony_source[ks], pred_colonies[kp], ks, is_primary=(output is primary_prob))

                    output[word] += acc
    elif scorer == "overlap":
        pass
    
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

    for meg, label_id, run_info in tqdm(run, desc="Validating", unit="window"):
        word = run.id_to_word[int(label_id)]
        label = normalize_word(word)
        
        if label not in PRIMARY_VOCAB_TO_ID and label not in MOSES_VOCAB_TO_ID:
            continue
    
        _, _, p, m = model(meg, "cnn")
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
        save_model(f"pnpl/models/model_run{i}.pt")
        
        print(f"Finished training on run {i}, validating:")
        
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
