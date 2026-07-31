from pathlib import Path
import sys
import mne
mne.set_log_level("WARNING")
from mne.minimum_norm import prepare_inverse_operator
import pandas as pd
from tqdm import tqdm
from colony import Colony, build_hemisphere_mirror_map, compute_gain, load_subject, read_subject_record, setup_inverse, BANDS, DATASET_SPECS, TIMESTEP
import numpy as np
from collections import defaultdict

INTERACTIVE = sys.stderr.isatty()

reference_colonies = {
    "eegmmidb": [
        "task1_real_left_fist",
        "task1_real_right_fist",
        "task2_imagine_left_fist",
        "task2_imagine_right_fist",
        "task3_real_both_feet",
        "task3_real_both_fists",
        "task4_imagine_both_feet",
        "task4_imagine_both_fists"
    ]
}

test_eeg = {
    "eegmmidb": {
        "S011": [0, 1, 2, 3, 4, 5],
        "S012": [0, 1, 2, 3, 4, 5],
        "S013": [0, 1, 2, 3, 4, 5],
        "S014": [0, 1, 2, 3, 4, 5],
        "S015": [0, 1, 2, 3, 4, 5],
    }
}

WINDOW_LENGTH = 2000 / 1000

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
        
for band_name, band in BANDS.items():
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

        subject_confusion = defaultdict(int)
        subject_entries = []

        for r_id in record_indices:
            print(f"    Record: {r_id}")
            raw_record, _ = read_subject_record(dataset, subject_active_files[r_id])
            n_windows = int(raw_record.duration // WINDOW_LENGTH)
            
            for i in tqdm(range(n_windows), desc=f"{r_id}", leave=False, disable=not INTERACTIVE):
                print(f"        Window: {i+1}/{n_windows}")
                start_time = i * WINDOW_LENGTH
                end_time = start_time + WINDOW_LENGTH
            
                band_data = defaultdict(dict)

                true_event = get_window_event(raw_record, start_time, end_time, spec) or "rest"
                raw_window = raw_record.copy().crop(tmin=start_time, tmax=end_time)

                for band_name, band in tqdm(BANDS.items(), desc=f"{subject} bands", leave=False, disable=not INTERACTIVE):
                    print(f"          Band: {band_name}")
                    low = band["low"]
                    high = min(band["high"], raw_baseline.info["sfreq"] / 2 - 1)
                    
                    raw_filtered = raw_window.copy()
                    raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4)

                    colonies = compute_gain(prepared_inv, raw_filtered, inverse_mirror_map, lambda2, TIMESTEP,
                                            include_vol=True, include_csd=True, include_inverse=True,
                                            include_pos=True, include_neg=True,
                                            use_epochs=False, mirror=True)

                    for (source, _), colony in colonies.items():
                        if source not in band_data:
                            band_data[source] = {}
                        band_data[source][band_name] = colony

            
                def get_source_overlap(source):
                    overlaps = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
                    distance = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

                    for band_name, colony in band_data[source].items():
                        pos_weights = colony.pos_weights()
                        neg_weights = colony.neg_weights()
                        pos_threshold = np.quantile(pos_weights, 0.75)
                        neg_threshold = np.quantile(neg_weights, 0.75)
                        pos_top = set(np.where(pos_weights >= pos_threshold)[0])
                        neg_top = set(np.where(neg_weights >= neg_threshold)[0])

                        for ref, ref_colony in ref_cache[source][band_name].items():
                            ref_pos_weights = ref_colony.pos_weights()
                            ref_neg_weights = ref_colony.neg_weights()
                            ref_pos_top = set(np.where(ref_pos_weights >= np.quantile(ref_pos_weights, 0.75))[0])
                            ref_neg_top = set(np.where(ref_neg_weights >= np.quantile(ref_neg_weights, 0.75))[0])

                            pos_union = pos_top | ref_pos_top
                            neg_union = neg_top | ref_neg_top
                            overlaps[ref]["pos"][band_name] = len(pos_top & ref_pos_top) / len(pos_union) if pos_union else 0.0
                            overlaps[ref]["neg"][band_name] = len(neg_top & ref_neg_top) / len(neg_union) if neg_union else 0.0
                            distance[ref]["pos"][band_name] = np.sqrt(np.sum(pos_weights * (pos_weights - ref_pos_weights) ** 2))
                            distance[ref]["neg"][band_name] = np.sqrt(np.sum(neg_weights * (neg_weights - ref_neg_weights) ** 2))

                    return overlaps, distance
                
                source_results = {}
                for source_type in ["csd", "inverse"]:
                    if source_type in band_data:
                        source_results[source_type] = get_source_overlap(source_type)

                best_score = -1
                best_ref = None

                for ref in all_refs:
                    score = 0.0
                    for source_type, (overlaps, _) in source_results.items():
                        score += np.mean(list(overlaps[ref]["pos"].values()))
                        score += np.mean(list(overlaps[ref]["neg"].values()))

                    if score > best_score:
                        best_score = score
                        best_ref = ref

                best_distance = 0.0
                for source_type, (_, distances) in source_results.items():
                    best_distance += sum(distances[best_ref]["pos"].values())
                    best_distance += sum(distances[best_ref]["neg"].values())

                subject_confusion[(true_event, best_ref)] += 1
                dataset_confusion[(true_event, best_ref)] += 1
                subject_entries.append({
                    "true": true_event,
                    "predicted": best_ref,
                    "correct": true_event == best_ref,
                    "overlap": best_score,
                    "distance": best_distance,
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
                overlaps = [e["overlap"] for e in subset]
                dists = [e["distance"] for e in subset]
                print(f"  {label}: overlap={np.mean(overlaps):.4f}±{np.std(overlaps):.4f}"
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
            overlaps = [e["overlap"] for e in subset]
            dists = [e["distance"] for e in subset]
            print(f"  {label}: overlap={np.mean(overlaps):.4f}±{np.std(overlaps):.4f}"
                  f"  dist={np.mean(dists):.4f}±{np.std(dists):.4f}")

    print(f"\n  Per-subject:")
    for subject, entries in subject_results.items():
        if entries:
            c = sum(1 for e in entries if e["correct"])
            t = len(entries)
            print(f"    {subject}: {c}/{t} ({c/t:.1%})")
