from mne.minimum_norm import prepare_inverse_operator, apply_inverse_raw, write_inverse_operator, read_inverse_operator, make_inverse_operator
import mne
import sys
from scipy.spatial import KDTree
import scipy
import numpy as np
import collections
import itertools
from pathlib import Path
import keras
from hmmlearn import hmm
import torch
from umap.parametric_umap import ParametricUMAP
from sklearn.decomposition import IncrementalPCA
from colony import DATASET_SPECS, BANDS, load_subject, read_subject_record, setup_inverse
from denstream import DenStream

SEED = 42
UMAP_COMPONENTS = 100
PCA_COMPONENTS = 1000
TIMESTEPS = [50, 100, 200] # ms
CLUSTER_WINDOWS = [3, 5, 8, 15]

keras.utils.set_random_seed(SEED) # seed for UMAP encoder

def collect(data: np.ndarray, step: int, sfreq: int):
    raw_growth = np.zeros((data.shape[0], data.shape[1] // step))
    pos_growth = np.zeros((data.shape[0], data.shape[1] // step))
    neg_growth = np.zeros((data.shape[0], data.shape[1] // step))
    
    a = np.abs(scipy.signal.hilbert(data))
    b = scipy.ndimage.uniform_filter1d(a, size=int(2.0 * sfreq), axis=1)
    data = (a-b)/b
        
    last_matrix = np.nan_to_num(data[:, 0:step].mean(axis=1), nan=0.0)
    for i in np.arange(1, np.floor(data.shape[1] // step)):
        start_time = int(i * step)
        end_time = int(min((i + 1) * step, data.shape[1]))
        current_matrix = np.nan_to_num(data[:, start_time:end_time].mean(axis=1), nan=0.0)
        raw_growth[:, int(i)] = current_matrix - last_matrix
        pos_growth[:, int(i)] = np.maximum(current_matrix - last_matrix, 0)
        neg_growth[:, int(i)] = np.minimum(current_matrix - last_matrix, 0)
        last_matrix = current_matrix
    
    return raw_growth, pos_growth, neg_growth

def cluster(train_samples):
    model = DenStream(
        epsilon=0.3,      # neighborhood radius for micro-clusters
        beta=0.2,         # outlier threshold (micro-cluster weight below beta*mu = outlier)
        mu=10,            # minimum weight for a micro-cluster to be "potential" (real)
        lambd=0.001,      # decay factor (set near-zero since your data isn't temporal priority)
        min_samples=5     # minimum points for DBSCAN in the macro-clustering step
    )
    
    model.fit_generator(train_samples)
    return model

def predict(model: DenStream, points: np.ndarray):
    labels = model._request_clustering()
    if len(labels) == 0:
        return np.full(points.shape[0], -1, dtype=np.int32)
    centers = np.concatenate([c.center for c in model.p_micro_clusters], axis=0)
    closest = np.argmin(np.linalg.norm(points[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2), axis=1)
    return labels[closest]

def generate_sphere_centers(coordinates: np.ndarray, r: float, k: int):
    tree = KDTree(coordinates)
    center = coordinates.mean(axis=0)
    step = r / 2

    visited = set()
    queue = collections.deque()

    def grid_key(point):
        return tuple(np.round(point / step).astype(int))

    start_key = grid_key(center)
    visited.add(start_key)
    queue.append(center)

    centers = []
    indices = []

    while queue:
        point = queue.popleft()
        if len(tree.query_ball_point(point, r)) == 0:
            continue

        _, knn_idx = tree.query(point, k=k)
        centers.append(point)
        indices.append(knn_idx)

        for axis in range(3):
            for direction in (-1, 1):
                neighbor = point.copy()
                neighbor[axis] += direction * step
                key = grid_key(neighbor)
                if key not in visited:
                    visited.add(key)
                    queue.append(neighbor)

    return np.array(centers), np.array(indices)

def generate_subject_data():
    target_subjects = {
        "eegmmidb": ["S001", "S002", "S003", "S004", "S005", "S006", "S007", "S008", "S009", "S010"],
    }

    for dataset, subjects in target_subjects.items():
        print(f"Processing dataset: {dataset}")
        spec = DATASET_SPECS[dataset]

        for subject in subjects:
            print(f"Processing subject: {subject}")
            subject_path = Path("growth") / dataset / subject
            subject_path.mkdir(parents=True, exist_ok=True)
            sub_baseline, sub_active = load_subject(dataset, subject)
            raw_baseline, _ = read_subject_record(dataset, sub_baseline)

            inv, src, bem = setup_inverse(dataset, subject, raw_baseline)
            snr = 3.0
            lambda2 = 1.0 / (snr ** 2)
            prepared_inv = prepare_inverse_operator(inv, nave=1, lambda2=lambda2)

            src_data = mne.read_source_spaces(src)
            lh_vertno = inv['src'][0]['vertno']
            rh_vertno = inv['src'][1]['vertno']
            all_xyz_coordinates = np.vstack([src_data[0]['rr'][lh_vertno], src_data[1]['rr'][rh_vertno]])
            np.save(subject_path / "coordinates.npy", all_xyz_coordinates)

            target_records = sub_active[:8]
            for record_index, record in enumerate(target_records):
                print(f"  Record: {record_index + 1}/{len(target_records)}")
                raw_active, _ = read_subject_record(dataset, record)
                sfreq = int(raw_active.info["sfreq"])

                grouped_annotations = {}
                for annotation in raw_active.annotations:
                    desc = int(annotation["description"].item().strip())
                    key = spec["event_ids"][desc]
                    if key not in grouped_annotations:
                        grouped_annotations[key] = []
                    grouped_annotations[key].append((annotation["onset"].item(), annotation["onset"].item() + annotation["duration"].item()))

                for band_name, band in BANDS.items():
                    print(f"    Band: {band_name}")
                    low = band["low"]
                    high = min(band["high"], sfreq / 2 - 1)
                    raw_filtered = raw_active.copy()
                    raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4)

                    stc = apply_inverse_raw(
                        raw_filtered, prepared_inv,
                        lambda2=lambda2, buffer_size=5000,
                        method="dSPM", prepared=True
                    )
                    stc_data = stc.data

                    for timestep, cluster_window in itertools.product(TIMESTEPS, CLUSTER_WINDOWS):
                        timestep_ms = timestep / 1000
                        for group, annotations in grouped_annotations.items():
                            output_path = subject_path / band_name / f"t{timestep}c{cluster_window}" / group / f"record_{record_index}.npy"
                            output_path.parent.mkdir(parents=True, exist_ok=True)

                            raws = []
                            poss = []
                            negs = []

                            largest_end_index = int(max((et - st) for st, et in annotations) * sfreq)
                            for start_time, end_time in annotations:
                                sample = stc_data[:, int(start_time * sfreq):int(end_time * sfreq)]

                                if sample.shape[1] < largest_end_index:
                                    padding = np.zeros((sample.shape[0], largest_end_index - sample.shape[1]))
                                    sample = np.hstack([sample, padding])

                                raw_growth, pos_growth, neg_growth = collect(sample, step=int(timestep_ms * sfreq), sfreq=sfreq)
                                n_full_windows = raw_growth.shape[1] // cluster_window
                                trim = n_full_windows * cluster_window
                                raw_split = np.split(raw_growth[:, :trim], n_full_windows, axis=1)
                                pos_split = np.split(pos_growth[:, :trim], n_full_windows, axis=1)
                                neg_split = np.split(neg_growth[:, :trim], n_full_windows, axis=1)

                                raws.extend(raw_split)
                                poss.extend(pos_split)
                                negs.extend(neg_split)

                            np.save(output_path, np.array([np.stack(raws), np.stack(poss), np.stack(negs)]))

def generate_models(for_timestep: int, for_cluster_window: int):
    pass

if __name__ == "__main__":
    generate_subject_data()

# + gotta fix the inhomogenous arrays (presumably due to different sized events)
# gotta do a bunch of aggregation types (same subject, same event, same dataset...) into the HMM
# + gotta select equidistant, all-encompassing spheres/vertex centers instead of random
# gotta be able to visualize the clusters in their spheres (approximated? export with coordinates?)

# one thing: are we ever going to take bands into account here? probably should, right?
# oh boy... how are we going to mass run all of these?
# so, we need to have a loader to get all or some (random sample?) of the growth/ data to put into the model
# once the model is fitted on all of that, we can then 1) check the clusters in a 3d view/animation (cause it's over time)
# then, we can do inference on either new or same data to determine what clusters fall under events
# then, IF the events have meaningfully common/different events from each other AND from the relaxation/non-event states
    # then we can test those events (weighted? linear combinations) for discrimination between events and get probabilities
# also, we need some way to keep track of localization of clusters for event discrimination (a sphere in the temporal vs in the parietal for ex)
# and everything should be invariant to dataset/subjects (new datasets should have resting periods)