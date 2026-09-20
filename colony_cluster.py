from pathlib import Path
import sys
import mne
mne.set_log_level("WARNING")
from mne.minimum_norm import prepare_inverse_operator
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from colony import Colony, build_hemisphere_mirror_map, compute_gain, load_subject, read_subject_record, setup_inverse, BANDS, DATASET_SPECS, TIMESTEP
import numpy as np
from collections import defaultdict

INTERACTIVE = sys.stderr.isatty()

reference_colonies = {
    "grasplift": [
        "HandStart",
        "FirstDigitTouch",
        "LiftOff",
        # "Replace",
    ]
}

test_eeg = {
    "grasplift": {
        "subj1": [0, 1, 2, 3],
        #"subj2": [0, 1, 2, 3, 4, 5],
        #"subj9": [0, 1, 2, 3, 4, 5],
        #"subj10": [0, 1, 2, 3, 4, 5],
        #"subj11": [0, 1, 2, 3, 4, 5],
        "subj12": [0, 1, 2, 3],
    }
}

WINDOW_LENGTH = 1000 / 1000

PERCENTILES = [(0.5, 0.99), (0.75, 0.99), (0.85, 0.99), (0.95, 0.99), (0.5, 0.95), (0.75, 0.95), (0.85, 0.95), (0.5, 0.85)]

TARGET_BANDS = {k: v for k, v in BANDS.items() if k not in ["standard", "whole"]}
# at the moment, standard and whole are too sweeping

def get_window_event(raw, start_time, end_time, spec):
    for annotation in raw.annotations:
        onset = annotation["onset"].item()
        dur = annotation["duration"].item()
        ann_end = onset + dur
        overlap = min(end_time, ann_end) - max(start_time, onset)
        
        if overlap > (end_time - start_time) * 0.5:
            desc = int(annotation["description"].item().strip())
            return spec["event_ids"].get(desc)
    
    return None

ref_cache = defaultdict(lambda: defaultdict(dict))
all_refs = set(ref for refs in reference_colonies.values() for ref in refs)
        
for band_name, band in TARGET_BANDS.items():
    for source_type in ["vol", "csd", "inverse"]:
        for ref_dataset, refs in reference_colonies.items():
            for ref in refs:
                pos_path = Path(f"coalesce/{ref_dataset}/{source_type}/{band_name}/{ref}/pos.csv")
                neg_path = Path(f"coalesce/{ref_dataset}/{source_type}/{band_name}/{ref}/neg.csv")
                
                pos_df = pd.read_csv(pos_path)
                neg_df = pd.read_csv(neg_path)
                pos_values = pos_df["value"].values
                neg_values = neg_df["value"].values
                cluster_colony = Colony(len(pos_df), include_pos=True, include_neg=True)
                cluster_colony.colony_pos = pos_values
                cluster_colony.colony_neg = neg_values
                
                ref_cache[source_type][band_name][ref] = cluster_colony

for dataset, subjects in test_eeg.items():
    spec = DATASET_SPECS[dataset]

    dataset_confusion = defaultdict(int)
    subject_results = {}
    all_subjects_band_data = defaultdict(lambda: defaultdict(tuple[dict, int]))

    for subject, record_indices in tqdm(subjects.items(), desc="Subjects", disable=not INTERACTIVE):
        print(f"Subject: {subject}")
        subject_baseline_file, subject_active_files = load_subject(dataset, subject)
        raw_baseline, _ = read_subject_record(dataset, subject_baseline_file)

        inv, src, bem = setup_inverse(dataset, subject, raw_baseline)
        snr = 3.0
        lambda2 = 1.0 / (snr ** 2)
        prepared_inv = prepare_inverse_operator(
            inv,
            nave=1,
            lambda2=lambda2
        )

        src_data = mne.read_source_spaces(src)

        lh_vertno = inv['src'][0]['vertno']
        rh_vertno = inv['src'][1]['vertno']
        inverse_mirror_map = build_hemisphere_mirror_map(src_data, lh_vertno, rh_vertno)

        for r_id in record_indices:
            print(f"    Record: {r_id}")
            raw_record, _ = read_subject_record(dataset, subject_active_files[r_id])
            n_windows = int(raw_record.duration // WINDOW_LENGTH)
            
            filtered_records = {}
            for band_name, band in TARGET_BANDS.items():
                print(f"    Filtering band: {band_name}")
                low = band["low"]
                high = min(band["high"], raw_baseline.info["sfreq"] / 2 - 1)
                filtered = raw_record.copy()
                filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4)
                filtered_records[band_name] = filtered

            for i in tqdm(range(n_windows), desc=f"{r_id}", leave=False, disable=not INTERACTIVE):
                print(f"        Window: {i+1}/{n_windows}")
                start_time = i * WINDOW_LENGTH
                end_time = start_time + WINDOW_LENGTH
                
                if end_time > raw_record.times[-1]:
                    break
                
                true_event = get_window_event(raw_record, start_time, end_time, spec)
                if not true_event:
                    continue
                
                for band_name in TARGET_BANDS:
                    raw_window = filtered_records[band_name].copy().crop(tmin=start_time, tmax=end_time)

                    colonies = compute_gain(prepared_inv, raw_window, lambda2, TIMESTEP, inverse_mirror_map,
                                            include_vol=True, include_csd=True, include_inverse=True,
                                            include_pos=True, include_neg=True,
                                            use_epochs=False)

                    for (source, _), colony in colonies.items():
                        # need to make this temporal based so we can differentiate bteween different time windows
                        # (source, subject, epoch_id) => (list_of_band_colonies, true_event)
                        
                        # yeah broski we need to rewrite this file and actually decide what we're clustering on
                        
                        l = all_subjects_band_data[subject][source]
                        if not l:
                            l = ({}, true_event)
                            all_subjects_band_data[subject][source] = l

                        l[0][band_name] = colony
    
    source_band_pos_weights = {}
    source_band_neg_weights = {}
    
    for subject, subject_data in all_subjects_band_data.items():
        for source_type, (bands, true_event) in subject_data.items():
            for band_name, colony_data in bands.items():
                X_pos = []
                X_neg = []
                y = []
                
                for _, entries in colony_data.items():
                    for true_event, colony in entries:
                        pos_weights = colony.pos_weights()
                        neg_weights = colony.neg_weights()
                        X_pos.append(pos_weights)
                        X_neg.append(neg_weights)
                        y.append(true_event)
                    
                clf = LogisticRegression()
                clf.fit(X_pos, y)
                pos_weights = np.abs(clf.coef_)
                
                for event_idx, event_name in enumerate(clf.classes_):
                    source_band_pos_weights[(source_type, event_name, band_name)] = pos_weights[event_idx]

                clf.fit(X_neg, y)
                neg_weights = np.abs(clf.coef_)
                
                for event_idx, event_name in enumerate(clf.classes_):
                    source_band_neg_weights[(source_type, event_name, band_name)] = neg_weights[event_idx]
                
    for subject in subjects.keys():
        subject_confusion = defaultdict(int)
        subject_entries = []
                        
        def predict(source, band_colonies):
            overlaps = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
            distance = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

            for band_name, colony in band_colonies.items():
                if band_name in ["standard", "whole"]:
                    continue
                    
                pos_weights = colony.pos_weights()
                neg_weights = colony.neg_weights()

                for ref, ref_colony in ref_cache[source][band_name].items():
                    ref_pos_weights = ref_colony.pos_weights()
                    ref_neg_weights = ref_colony.neg_weights()

                    for pstart, pend in PERCENTILES:
                        pkey = (pstart, pend)

                        pos_lo, pos_hi = np.quantile(pos_weights, pstart), np.quantile(pos_weights, pend)
                        neg_lo, neg_hi = np.quantile(neg_weights, 1 - pstart), np.quantile(neg_weights, 1 - pend)
                        pos_top = set(np.where((pos_weights >= pos_lo) & (pos_weights <= pos_hi))[0])
                        neg_top = set(np.where((neg_weights >= neg_lo) & (neg_weights <= neg_hi))[0])

                        ref_pos_lo, ref_pos_hi = np.quantile(ref_pos_weights, pstart), np.quantile(ref_pos_weights, pend)
                        ref_neg_lo, ref_neg_hi = np.quantile(ref_neg_weights, 1 - pend), np.quantile(ref_neg_weights, 1 - pstart)
                        ref_pos_top = set(np.where((ref_pos_weights >= ref_pos_lo) & (ref_pos_weights <= ref_pos_hi))[0])
                        ref_neg_top = set(np.where((ref_neg_weights >= ref_neg_lo) & (ref_neg_weights <= ref_neg_hi))[0])

                        pos_union = pos_top | ref_pos_top
                        neg_union = neg_top | ref_neg_top
                        overlaps[ref]["pos"][band_name][pkey] = len(pos_top & ref_pos_top) / len(pos_union) if pos_union else 0.0
                        overlaps[ref]["neg"][band_name][pkey] = len(neg_top & ref_neg_top) / len(neg_union) if neg_union else 0.0

                        pos_sel = list(pos_top | ref_pos_top)
                        neg_sel = list(neg_top | ref_neg_top)
                        distance[ref]["pos"][band_name][pkey] = np.sqrt(np.sum((pos_weights[pos_sel] - ref_pos_weights[pos_sel]) ** 2)) if pos_sel else 0.0
                        distance[ref]["neg"][band_name][pkey] = np.sqrt(np.sum((neg_weights[neg_sel] - ref_neg_weights[neg_sel]) ** 2)) if neg_sel else 0.0

            return overlaps, distance
        
        source_results = {}
        # how do we handle different CSD electrode arrays (sizes)?
        for source_type in ["inverse"]: #["csd", "inverse"]:
            bands, te = all_subjects_band_data[subject][source_type]
            source_results[source_type] = (predict(source_type, bands), te)

        best_score = -1
        best_ref = None

        for ref in all_refs:
            score = 0.0
            for source_type, (overlaps, _) in source_results.items():
                for sign in ["pos", "neg"]:
                    for band_vals in overlaps[ref][sign].values():
                        for pkey in PERCENTILES:
                            score += band_vals[pkey]

            if score > best_score:
                best_score = score
                best_ref = ref

        pct_metrics = {}
        for pkey in PERCENTILES:
            pct_overlap = 0.0
            pct_distance = 0.0
            for source_type, (overlaps, distances) in source_results.items():
                for sign in ["pos", "neg"]:
                    band_overlaps = [overlaps[best_ref][sign][bn][pkey] for bn in overlaps[best_ref][sign]]
                    band_dists = [distances[best_ref][sign][bn][pkey] for bn in distances[best_ref][sign]]
                    pct_overlap += np.mean(band_overlaps) if band_overlaps else 0.0
                    pct_distance += sum(band_dists)
            pct_metrics[pkey] = {"overlap": pct_overlap, "distance": pct_distance}

        subject_confusion[(true_event, best_ref)] += 1
        dataset_confusion[(true_event, best_ref)] += 1
        subject_entries.append({
            "true": true_event,
            "predicted": best_ref,
            "correct": true_event == best_ref,
            "percentiles": pct_metrics,
        })

        subject_results[subject] = subject_entries

        total = sum(subject_confusion.values())
        correct = sum(v for (t, p), v in subject_confusion.items() if t == p)
        acc = correct / total if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"Subject: {subject}  —  {correct}/{total} ({acc:.1%})")
        print(f"{'='*60}")

        for true_event in sorted({t for t, _ in subject_confusion}):
            preds = {p: v for (t, p), v in subject_confusion.items() if t == true_event}
            row_total = sum(preds.values())
            row_correct = preds.get(true_event, 0)
            pred_str = "  ".join(f"{p}={v}" for p, v in sorted(preds.items(), key=lambda x: -x[1]))
            marker = "*" if true_event in all_refs else " "
            print(f"  {marker}{true_event}: {row_correct}/{row_total}  [{pred_str}]")

        for label, filt in [("correct", True), ("wrong", False)]:
            subset = [e for e in subject_entries if e["correct"] == filt]
            if subset:
                print(f"  {label}:")
                for pkey in PERCENTILES:
                    overlaps = [e["percentiles"][pkey]["overlap"] for e in subset]
                    dists = [e["percentiles"][pkey]["distance"] for e in subset]
                    print(f"    {pkey[0]:.0%}-{pkey[1]:.0%}: overlap={np.mean(overlaps):.4f}±{np.std(overlaps):.4f}"
                          f"  dist={np.mean(dists):.4f}±{np.std(dists):.4f}")

    total = sum(dataset_confusion.values())
    correct = sum(v for (t, p), v in dataset_confusion.items() if t == p)
    acc = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}  —  {correct}/{total} ({acc:.1%})")
    print(f"{'='*60}")

    for true_event in sorted({t for t, _ in dataset_confusion}):
        preds = {p: v for (t, p), v in dataset_confusion.items() if t == true_event}
        row_total = sum(preds.values())
        row_correct = preds.get(true_event, 0)
        pred_str = "  ".join(f"{p}={v}" for p, v in sorted(preds.items(), key=lambda x: -x[1]))
        marker = "*" if true_event in all_refs else " "
        print(f"  {marker}{true_event}: {row_correct}/{row_total}  [{pred_str}]")

    all_entries = [e for subj in subject_results.values() for e in subj]
    for label, filt in [("correct", True), ("wrong", False)]:
        subset = [e for e in all_entries if e["correct"] == filt]
        if subset:
            print(f"  {label}:")
            for pkey in PERCENTILES:
                overlaps = [e["percentiles"][pkey]["overlap"] for e in subset]
                dists = [e["percentiles"][pkey]["distance"] for e in subset]
                print(f"    {pkey[0]:.0%}-{pkey[1]:.0%}: overlap={np.mean(overlaps):.4f}±{np.std(overlaps):.4f}"
                      f"  dist={np.mean(dists):.4f}±{np.std(dists):.4f}")

    print(f"\n  Per-subject:")
    for subject, entries in subject_results.items():
        if entries:
            c = sum(1 for e in entries if e["correct"])
            t = len(entries)
            print(f"    {subject}: {c}/{t} ({c/t:.1%})")




# I don't even fucking know at this point; we just gotta go for maximum power
    # + maybe the scoring needs to take all bands + CSD into account (voltage is useless, drop it)
    # + also implement the negatives as well
    # also need to start exporting the things that are failing so we have an idea of what's failing and why
        # maybe have a 3D plot of what is failing based on the intersect/union
