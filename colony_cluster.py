from pathlib import Path
import mne
from mne.minimum_norm import prepare_inverse_operator
import pandas as pd
from colony import build_hemisphere_mirror_map, compute_gain, load_subject, read_subject_record, setup_inverse, BANDS, DATASET_SPECS, TIMESTEP
import numpy as np

reference_colonies = [
    "task1_real_left_fist",
    "task2_imagine_left_fist",
    "task3_real_both_fists",
    "task4_imagine_both_fists"
]

test_eeg = {
    "eegmmidb": {
        "S001": [0, 1, 2, 3, 4, 5],
        "S002": [0, 1, 2, 3, 4, 5],
        "S003": [0, 1, 2, 3, 4, 5],
        "S004": [0, 1, 2, 3, 4, 5],
        "S005": [0, 1, 2, 3, 4, 5],
    }
}

WINDOW_LENGTH = 2000 / 1000

for dataset, subjects in test_eeg.items():
    spec = DATASET_SPECS[dataset]
    subject_stats = {}

    for subject, record_indices in subjects.items():
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

        distances = {ref: [] for ref in reference_colonies}

        for band_name, band in BANDS.items():
            low = band["low"]
            high = min(band["high"], raw_baseline.info["sfreq"] / 2 - 1)

            for r_id in record_indices:
                raw_record, _ = read_subject_record(dataset, subject_active_files[r_id])
                raw_filtered = raw_record.copy()
                raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4)

                for i in range(int(raw_filtered.duration // WINDOW_LENGTH)):
                    start_time = i * WINDOW_LENGTH
                    end_time = start_time + WINDOW_LENGTH
                    raw_window = raw_filtered.copy().crop(tmin=start_time, tmax=end_time)

                    colonies = compute_gain(prepared_inv, raw_window, inverse_mirror_map, lambda2, TIMESTEP,
                                            include_vol=True, include_csd=True, include_inverse=True,
                                            include_pos=True, include_neg=True,
                                            use_epochs=False)

                    for (source, group), colony in colonies.items():
                        for ref in reference_colonies:
                            cluster_path = Path(f"coalesce/{dataset}/{source}/{band_name}/{ref}/pos.csv")
                            if not cluster_path.exists():
                                continue
                            cluster_df = pd.read_csv(cluster_path)
                            cluster_values = cluster_df["value"].values

                            weights = colony.pos_weights()
                            distance = np.sqrt(np.sum(weights * (cluster_values - colony.colony_pos) ** 2))
                            distances[ref].append({
                                "source": source,
                                "band": band_name,
                                "record": r_id,
                                "window": i,
                                "distance": distance,
                            })

        subject_stats[subject] = distances

        print(f"\n{'='*60}")
        print(f"Subject: {subject}")
        print(f"{'='*60}")
        for ref, entries in distances.items():
            if not entries:
                continue
            dists = [e["distance"] for e in entries]
            print(f"  {ref}:")
            print(f"    n={len(dists)}  mean={np.mean(dists):.4f}  std={np.std(dists):.4f}  min={np.min(dists):.4f}  max={np.max(dists):.4f}")

    print(f"\n{'='*60}")
    print(f"Dataset summary: {dataset}")
    print(f"{'='*60}")
    for ref in reference_colonies:
        all_dists = []
        for subject, distances in subject_stats.items():
            all_dists.extend(e["distance"] for e in distances[ref])
        if not all_dists:
            continue
        print(f"  {ref}:")
        print(f"    n={len(all_dists)}  mean={np.mean(all_dists):.4f}  std={np.std(all_dists):.4f}  min={np.min(all_dists):.4f}  max={np.max(all_dists):.4f}")
        for subject, distances in subject_stats.items():
            subj_dists = [e["distance"] for e in distances[ref]]
            if subj_dists:
                print(f"      {subject}: mean={np.mean(subj_dists):.4f}  std={np.std(subj_dists):.4f}")
