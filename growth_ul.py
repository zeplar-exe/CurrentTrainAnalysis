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
from umap.parametric_umap import ParametricUMAP
from sklearn.decomposition import IncrementalPCA
from colony import DATASET_SPECS, load_subject, read_subject_record, setup_inverse

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

def cluster(cluster_samples):
    pca = IncrementalPCA(n_components=PCA_COMPONENTS) #random_state=SEED
    reduced = None
    sample_length = None
    
    for sample in cluster_samples:
        if not sample_length:
            sample_length = sample.shape[1]
        elif sample_length != sample.shape[1]:
            raise ValueError(f"All samples must have the same length, but got {sample_length} and {sample.shape[1]}")
        pca.partial_fit(sample)
    
    for sample in cluster_samples:
        if reduced is None:
            reduced = pca.transform(sample).astype(np.float32)
        else:
            reduced = np.hstack([reduced, pca.transform(sample).astype(np.float32)])
    
    print(f"PCA: {PCA_COMPONENTS} components retain {pca.explained_variance_ratio_.sum():.2%} of variance")
    
    states = 50
    model = hmm.GaussianHMM(
        n_components=states,        # number of hidden states (sweep this)
        covariance_type="diag", # "full" is richer but needs more data per state
        n_iter=200,            # EM iterations
        random_state=SEED
    )
    model.fit(reduced, lengths=[sample_length] * len(cluster_samples))
    
    templates = model.means_
    states = model.predict(reduced)
    transitions = model.transmat_
    
    log_likelihood = model.score(reduced)
    n_params = states * 200 + states * 200 + states * states  # means + covars(diag) + transitions
    bic = -2 * log_likelihood * len(reduced) + n_params * np.log(len(reduced))

def generate_sphere_centers(coordinates: np.ndarray, r: float):
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

    while queue:
        point = queue.popleft()
        neighbors = tree.query_ball_point(point, r)
        if len(neighbors) == 0:
            continue

        centers.append(point)

        for axis in range(3):
            for direction in (-1, 1):
                neighbor = point.copy()
                neighbor[axis] += direction * step
                key = grid_key(neighbor)
                if key not in visited:
                    visited.add(key)
                    queue.append(neighbor)

    return np.array(centers)

def generate_subject_data():
    target_subjects = {
        "eegmmidb": ["S001", "S002", "S003", "S004", "S005", "S006", "S007", "S008", "S009", "S010"][6:],
    }

    for dataset, subjects in target_subjects.items():
        print(f"Processing dataset: {dataset}")
        for subject in subjects:
            print(f"Processing subject: {subject}")
            subject_path = Path("growth") / dataset / subject
            sub_baseline, sub_active = load_subject(dataset, subject)
            raw_baseline, _ = read_subject_record(dataset, sub_baseline)

            inv, src, bem = setup_inverse("eegmmidb", subject, raw_baseline)
            snr = 3.0
            prepared_inv = prepare_inverse_operator(
                inv, 
                nave=1, 
                lambda2=1.0 / (snr ** 2)
            )
            
            src_data = mne.read_source_spaces(src)
            
            lh_vertno = 0
            rh_vertno = 0

            target_records = sub_active[:8]
            for record_index, record in enumerate(target_records):
                print(f"Processing record: {record_index + 1}/{len(target_records)}")
                raw_active, events_active = read_subject_record("eegmmidb", record)
                sfreq = int(raw_active.info["sfreq"])
                
                grouped_annotations = {}
                annotations_active = raw_active.annotations

                for annotation in annotations_active:
                    desc = int(annotation["description"].item().strip())
                    key = DATASET_SPECS["eegmmidb"]["event_ids"][desc]
                    
                    if key not in grouped_annotations:
                        grouped_annotations[key] = []
                    grouped_annotations[key].append((annotation["onset"].item(), annotation["onset"].item() + annotation["duration"].item()))
                    
                stc = apply_inverse_raw(
                    raw_active, 
                    prepared_inv, 
                    lambda2=1.0 / (snr ** 2),
                    buffer_size=5000,
                    method="dSPM",
                    prepared=True
                )
                stc_data = stc.data
                
                lh_vertno = stc.lh_vertno
                rh_vertno = stc.rh_vertno
                
                lh_coordinates = src_data[0]['rr'][lh_vertno]
                rh_coordinates = src_data[1]['rr'][rh_vertno]

                all_xyz_coordinates = np.vstack([lh_coordinates, rh_coordinates])
                
                for timestep, cluster_window in itertools.product(TIMESTEPS, CLUSTER_WINDOWS):
                    print(f"Processing timestep-cluster window: {timestep}-{cluster_window}")
                    timestep_ms = timestep / 1000
                    for group, annotations in grouped_annotations.items():
                        output_path = subject_path / f"t{timestep}c{cluster_window}" / group / f"record_{record_index}.npy"
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
                
                np.save(subject_path / "coordinates.npy", all_xyz_coordinates)

def generate_models(for_timestep: int, for_cluster_window: int):
    pass

if __name__ == "__main__":
    generate_subject_data()

# + gotta fix the inhomogenous arrays (presumably due to different sized events)
# gotta do a bunch of aggregation types (same subject, same event, same dataset...) into the HMM
# gotta figure out how many states is useful in the HMM (score sweeping?)
# + gotta select equidistant, all-encompassing spheres/vertex centers instead of random
# gotta be able to visualize the HMM states in the brain