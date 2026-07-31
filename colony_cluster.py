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
        "task4_imagine_both_feet"
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

for dataset, subjects in test_eeg.items():
    spec = DATASET_SPECS[dataset]

    dataset_confusion = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
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

        subject_confusion = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        subject_entries = []

        for band_name, band in tqdm(BANDS.items(), desc=f"{subject} bands", leave=False, disable=not INTERACTIVE):
            print(f"  Band: {band_name}")
            low = band["low"]
            high = min(band["high"], raw_baseline.info["sfreq"] / 2 - 1)

            ref_cache = {}
            for source_type in ["vol", "csd", "inverse"]:
                for ref_dataset, refs in reference_colonies.items():
                    for ref in refs:
                        cluster_path = Path(f"coalesce/{ref_dataset}/{source_type}/{band_name}/{ref}/pos.csv")
                        
                        cluster_df = pd.read_csv(cluster_path)
                        cluster_values = cluster_df["value"].values
                        cluster_colony = Colony(len(cluster_values), include_pos=True)
                        cluster_colony.colony_pos = cluster_values
                        weights = cluster_colony.pos_weights()
                        threshold = np.quantile(weights, 0.75)
                        ref_cache[(source_type, ref)] = (weights, set(np.where(weights >= threshold)[0]))

            for r_id in record_indices:
                print(f"    Record: {r_id}")
                raw_record, _ = read_subject_record(dataset, subject_active_files[r_id])
                raw_filtered = raw_record.copy()
                raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4)

                n_windows = int(raw_filtered.duration // WINDOW_LENGTH)
                for i in tqdm(range(n_windows), desc=f"{band_name} r{r_id}", leave=False, disable=not INTERACTIVE):
                    print(f"      Window: {i+1}/{n_windows}")
                    start_time = i * WINDOW_LENGTH
                    end_time = start_time + WINDOW_LENGTH

                    true_event = get_window_event(raw_record, start_time, end_time, spec) or "rest"
                    raw_window = raw_filtered.copy().crop(tmin=start_time, tmax=end_time)

                    colonies = compute_gain(prepared_inv, raw_window, inverse_mirror_map, lambda2, TIMESTEP,
                                            include_vol=True, include_csd=True, include_inverse=True,
                                            include_pos=True, include_neg=True,
                                            use_epochs=False, mirror=True)

                    for (source, _), colony in colonies.items():
                        colony_weights = colony.pos_weights()
                        colony_threshold = np.quantile(colony_weights, 0.75)
                        colony_top = set(np.where(colony_weights >= colony_threshold)[0])

                        best_ref = None
                        best_overlap = -1.0
                        best_distance = np.inf

                        for ref_dataset, refs in reference_colonies.items():
                            for ref in refs:
                                cluster_weights, ref_top = ref_cache[(source, ref)]

                                overlap = len(ref_top & colony_top) / len(ref_top | colony_top) if ref_top | colony_top else 0.0
                                distance = np.sqrt(np.sum(cluster_weights * (cluster_weights - colony_weights) ** 2))

                                if overlap > best_overlap or (overlap == best_overlap and distance < best_distance):
                                    best_overlap = overlap
                                    best_distance = distance
                                    best_ref = ref

                        if best_ref is not None:
                            subject_confusion[source][band_name][(true_event, best_ref)] += 1
                            dataset_confusion[source][band_name][(true_event, best_ref)] += 1
                            subject_entries.append({
                                "source": source,
                                "band": band_name,
                                "true": true_event,
                                "predicted": best_ref,
                                "correct": true_event == best_ref,
                                "overlap": best_overlap,
                                "distance": best_distance,
                            })

        subject_results[subject] = subject_entries

        print(f"\n{'='*60}")
        print(f"Subject: {subject}")
        print(f"{'='*60}")

        for source in sorted(subject_confusion):
            print(f"\n  [{source}]")
            for band_name in sorted(subject_confusion[source]):
                confusion = subject_confusion[source][band_name]
                total = sum(confusion.values())
                correct = sum(v for (t, p), v in confusion.items() if t == p)
                acc = correct / total if total > 0 else 0
                print(f"    {band_name}: {correct}/{total} ({acc:.1%})")

                for true_event in sorted({t for t, _ in confusion}):
                    preds = {p: v for (t, p), v in confusion.items() if t == true_event}
                    row_total = sum(preds.values())
                    row_correct = preds.get(true_event, 0)
                    pred_str = "  ".join(f"{p}={v}" for p, v in sorted(preds.items(), key=lambda x: -x[1]))
                    marker = "*" if true_event in reference_colonies else " "
                    print(f"     {marker}{true_event}: {row_correct}/{row_total}  [{pred_str}]")

    print(f"\n{'='*60}")
    print(f"Dataset summary: {dataset}")
    print(f"{'='*60}")

    for source in sorted(dataset_confusion):
        print(f"\n  [{source}]")
        for band_name in sorted(dataset_confusion[source]):
            confusion = dataset_confusion[source][band_name]
            total = sum(confusion.values())
            correct = sum(v for (t, p), v in confusion.items() if t == p)
            acc = correct / total if total > 0 else 0
            
            print(f"    {band_name}: {correct}/{total} ({acc:.1%})")

            for true_event in sorted({t for t, _ in confusion}):
                preds = {p: v for (t, p), v in confusion.items() if t == true_event}
                row_total = sum(preds.values())
                row_correct = preds.get(true_event, 0)
                pred_str = "  ".join(f"{p}={v}" for p, v in sorted(preds.items(), key=lambda x: -x[1]))
                marker = "*" if true_event in reference_colonies else " "
                print(f"     {marker}{true_event}: {row_correct}/{row_total}  [{pred_str}]")

        all_entries = [e for subj in subject_results.values() for e in subj if e["source"] == source]
        
        if all_entries:
            for label, filt in [("correct", True), ("wrong", False)]:
                subset = [e for e in all_entries if e["correct"] == filt]
                if subset:
                    print(f"    {label}: overlap={np.mean([e['overlap'] for e in subset]):.4f}±{np.std([e['overlap'] for e in subset]):.4f}"
                          f"  dist={np.mean([e['distance'] for e in subset]):.4f}±{np.std([e['distance'] for e in subset]):.4f}")

    print(f"\n  Per-subject accuracy:")
    for subject, entries in subject_results.items():
        if entries:
            correct = sum(1 for e in entries if e["correct"])
            total = len(entries)
            print(f"    {subject}: {correct}/{total} ({correct/total:.1%})")
