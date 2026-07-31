from pathlib import Path
import mne
mne.set_log_level("WARNING")
from mne.minimum_norm import prepare_inverse_operator
import pandas as pd
from tqdm import tqdm
from colony import Colony, build_hemisphere_mirror_map, compute_gain, load_subject, read_subject_record, setup_inverse, BANDS, DATASET_SPECS, TIMESTEP
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

    for subject, record_indices in tqdm(subjects.items(), desc="Subjects"):
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

        for band_name, band in tqdm(BANDS.items(), desc=f"{subject} bands", leave=False):
            low = band["low"]
            high = min(band["high"], raw_baseline.info["sfreq"] / 2 - 1)

            for r_id in record_indices:
                raw_record, _ = read_subject_record(dataset, subject_active_files[r_id])
                raw_filtered = raw_record.copy()
                raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4)

                n_windows = int(raw_filtered.duration // WINDOW_LENGTH)
                for i in tqdm(range(n_windows), desc=f"{band_name} r{r_id}", leave=False):
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
                            cluster_colony = Colony(len(cluster_values), include_pos=True, include_neg=False)
                            cluster_colony.colony_pos = cluster_values

                            cluster_weights = cluster_colony.pos_weights()
                            colony_weights = colony.pos_weights()
                            
                            distance = np.sqrt(np.sum(cluster_weights * (cluster_weights - colony_weights) ** 2))
                            norm_product = np.linalg.norm(cluster_weights) * np.linalg.norm(colony_weights)
                            cosine_sim = np.dot(cluster_weights, colony_weights) / norm_product if norm_product > 0 else 0.0
                            distances[ref].append({
                                "source": source,
                                "band": band_name,
                                "record": r_id,
                                "window": i,
                                "distance": distance,
                                "cosine": cosine_sim,
                            })

        subject_stats[subject] = distances

        print(f"\n{'='*60}")
        print(f"Subject: {subject}")
        print(f"{'='*60}")
        for ref, entries in distances.items():
            if not entries:
                continue
            dists = [e["distance"] for e in entries]
            cosines = [e["cosine"] for e in entries]
            print(f"  {ref}:")
            print(f"    dist:   n={len(dists)}  mean={np.mean(dists):.4f}  std={np.std(dists):.4f}  min={np.min(dists):.4f}  max={np.max(dists):.4f}")
            print(f"    cosine: n={len(cosines)}  mean={np.mean(cosines):.4f}  std={np.std(cosines):.4f}  min={np.min(cosines):.4f}  max={np.max(cosines):.4f}")

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
        all_cosines = []
        for subject, distances in subject_stats.items():
            all_cosines.extend(e["cosine"] for e in distances[ref])
        print(f"    dist:   n={len(all_dists)}  mean={np.mean(all_dists):.4f}  std={np.std(all_dists):.4f}  min={np.min(all_dists):.4f}  max={np.max(all_dists):.4f}")
        print(f"    cosine: n={len(all_cosines)}  mean={np.mean(all_cosines):.4f}  std={np.std(all_cosines):.4f}  min={np.min(all_cosines):.4f}  max={np.max(all_cosines):.4f}")
        for subject, distances in subject_stats.items():
            subj_dists = [e["distance"] for e in distances[ref]]
            subj_cosines = [e["cosine"] for e in distances[ref]]
            if subj_dists:
                print(f"      {subject}: dist={np.mean(subj_dists):.4f}±{np.std(subj_dists):.4f}  cosine={np.mean(subj_cosines):.4f}±{np.std(subj_cosines):.4f}")
