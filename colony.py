from mne.minimum_norm import prepare_inverse_operator, apply_inverse_raw, apply_inverse_epochs, write_inverse_operator, read_inverse_operator, make_inverse_operator
import mne
from mne.minimum_norm import InverseOperator
import numpy as np
from numpy.typing import NDArray
import scipy
from scipy.spatial import KDTree
from pathlib import Path
from rich.live import Live
from rich.panel import Panel
from typing import Literal

Source = Literal["vol", "csd", "inverse"]
ColonyMap = dict[tuple[Source, str], "Colony"]

DATASET_SPECS = {
    "eegmmidb": {
        "root": Path("./datasets/eegmmidb"),
        "sfreq": 160.0,
        "subjects": [f"S{i:03d}" for i in range(1, 103 + 1)],
        "channels": [
            "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz",
            "AF4", "AF8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7",
            "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
            "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
        ],
        "event_ids": {
            1: "task1_relax",
            2: "task1_real_left_fist",
            3: "task1_real_right_fist",
            4: "task2_relax",
            5: "task2_imagine_left_fist",
            6: "task2_imagine_right_fist",
            7: "task3_relax",
            8: "task3_real_both_fists",
            9: "task3_real_both_feet",
            10: "task4_relax",
            11: "task4_imagine_both_fists",
            12: "task4_imagine_both_feet",
        },
        "ignore_events": [1, 4, 7, 10]
    },
}

TIMESTEP = 50 / 1000 # s

BANDS = {
    "whole": {
        "low": 1,
        "high": 100,
    },
    "standard": {
        "low": 1,
        "high": 30
    },
    "theta": {
        "low": 1,
        "high": 4,
    }, 
    "delta": {
        "low": 4,
        "high": 8,
    },
    "alpha": {
        "low": 8,
        "high": 13,
    },
    "beta": {
        "low": 13,
        "high": 30,
    },
    "gamma": {
        "low": 30,
        "high": 100,
    },
}


def get_dataset_spec(dataset: str):
    try:
        return DATASET_SPECS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc


class Colony:
    DEVIATION_RESOLUTION = 2.0 # s
    
    def __init__(self, size: int, include_raw: bool = False, include_abs: bool = False, include_pos: bool = False, include_neg: bool = False):
        self.include_raw = include_raw
        self.include_abs = include_abs
        self.include_pos = include_pos
        self.include_neg = include_neg
        self.colony_raw = np.zeros(size)
        self.colony_abs = np.zeros(size)
        self.colony_pos = np.zeros(size)
        self.colony_neg = np.zeros(size)
        
    def feed(self, data: np.ndarray, step: int, sfreq: int):
        a = np.abs(scipy.signal.hilbert(data))
        b = scipy.ndimage.uniform_filter1d(a, size=int(Colony.DEVIATION_RESOLUTION * sfreq), axis=1)
        data = (a-b)/b
            
        last_matrix = np.nan_to_num(data[:, 0:step].mean(axis=1), nan=0.0)
        for i in np.arange(1, np.floor(data.shape[1] / step)):
            start_time = int(i * step)
            end_time = int(min((i + 1) * step, data.shape[1]))
            current_matrix = np.nan_to_num(data[:, start_time:end_time].mean(axis=1), nan=0.0)
            if self.include_raw:
                self.colony_raw += current_matrix - last_matrix
            if self.include_abs:
                self.colony_abs += np.abs(current_matrix - last_matrix)
            if self.include_pos:
                self.colony_pos += np.maximum(current_matrix - last_matrix, 0)
            if self.include_neg:
                self.colony_neg += np.minimum(last_matrix - current_matrix, 0)
            last_matrix = current_matrix
    
    def raw_weights(self):
        return (self.colony_raw - np.min(self.colony_raw)) / np.percentile(self.colony_raw, 99)
    
    def abs_weights(self):
        return (self.colony_abs - np.min(self.colony_abs)) / np.percentile(self.colony_abs, 99)
    
    def pos_weights(self):
        return (self.colony_pos - np.min(self.colony_pos)) / np.percentile(self.colony_pos, 99)

    def neg_weights(self):
        return (self.colony_neg - np.min(self.colony_neg)) / np.percentile(self.colony_neg, 99)


def _read_eegmmidb_record(record: Path, spec: dict):
    signal_path = Path(record)
    annotation_path = signal_path.with_name(signal_path.name.replace("_SIG_", "_ANN_"))

    signal = np.atleast_2d(np.loadtxt(signal_path, delimiter=",", dtype=float))
    if signal.shape[1] != len(spec["channels"]):
        raise ValueError(
            f"Unexpected channel count in {signal_path}: expected {len(spec['channels'])}, got {signal.shape[1]}"
        )

    raw = mne.io.RawArray(signal.T * 1e-6, mne.create_info(spec["channels"], spec["sfreq"], ch_types="eeg"), verbose="error")
    raw.set_montage("standard_1005")
    raw.set_eeg_reference(projection=True)

    annotation_data = np.atleast_2d(np.loadtxt(annotation_path, delimiter=",", dtype=float))
    annotation_data = annotation_data[~np.isin(annotation_data[:, 0], spec.get("ignore_events", [])), :]
    onset = (annotation_data[:, 3] - 1) / spec["sfreq"]
    duration = (annotation_data[:, 4] - annotation_data[:, 3] + 1) / spec["sfreq"]
    description = annotation_data[:, 0].astype(int).astype(str)

    raw.set_annotations(mne.Annotations(onset=onset, duration=duration, description=description))
    events = np.column_stack(
        [
            annotation_data[:, 3].astype(int) - 1,
            np.zeros(len(annotation_data), dtype=int),
            annotation_data[:, 0].astype(int),
        ]
    )

    return raw, events

def load_subject(dataset, subject):
    spec = get_dataset_spec(dataset)

    if dataset == "eegmmidb":
        subject_id = subject[1:] if subject.startswith("S") else subject
        runs = sorted(spec["root"].glob(f"SUB_{subject_id}_SIG_*.csv"))
        if not runs:
            raise ValueError(f"Subject {subject} not found in dataset {dataset}")
        return runs[0], runs[2:]

    raise ValueError(f"Unsupported dataset: {dataset}")
    
def fix_raw(dataset, raw):
    raw.notch_filter(freqs=60.0, fir_design='firwin')
    
    ica = mne.preprocessing.ICA(n_components=0.995, method='fastica')
    ica.fit(raw.copy().filter(1, min(100, raw.info["sfreq"] / 2 - 1), fir_design='firwin'))
    muscle_idx, scores = ica.find_bads_muscle(raw)
    ica.exclude = muscle_idx
    raw = ica.apply(raw.copy())
    
    if dataset == "physionet":
        case_fixing_map = {
            'Fc5': 'FC5', 'Fc3': 'FC3', 'Fc1': 'FC1', 'Fcz': 'FCz', 'Fc2': 'FC2', 'Fc4': 'FC4', 'Fc6': 'FC6',
            'Cp5': 'CP5', 'Cp3': 'CP3', 'Cp1': 'CP1', 'Cpz': 'CPz', 'Cp2': 'CP2', 'Cp4': 'CP4', 'Cp6': 'CP6',
            'Af7': 'AF7', 'Af3': 'AF3', 'Afz': 'AFz', 'Af4': 'AF4', 'Af8': 'AF8',
            'Ft7': 'FT7', 'Ft8': 'FT8', 'Tp7': 'TP7', 'Tp8': 'TP8',
            'Po7': 'PO7', 'Po3': 'PO3', 'Poz': 'POz', 'Po4': 'PO4', 'Po8': 'PO8'
        }

        mapping = {name: name.rstrip('.') for name in raw.ch_names}
        raw.rename_channels(mapping)
        raw.rename_channels(case_fixing_map)
        raw.set_montage('standard_1005')
        raw.set_eeg_reference(projection=True)
        
    return raw

def read_subject_record(dataset, record):
    spec = get_dataset_spec(dataset)

    if dataset == "eegmmidb":
        raw, events = _read_eegmmidb_record(record, spec)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    fix_raw(dataset, raw)

    return raw, events
      
def build_hemisphere_mirror_map(src_data, lh_vertno, rh_vertno):
    lh_coords = src_data[0]['rr'][lh_vertno]
    rh_coords = src_data[1]['rr'][rh_vertno]
    n_lh = len(lh_coords)
    n_rh = len(rh_coords)

    lh_mirrored = lh_coords.copy()
    lh_mirrored[:, 0] *= -1
    rh_mirrored = rh_coords.copy()
    rh_mirrored[:, 0] *= -1

    rh_tree = KDTree(rh_coords)
    lh_tree = KDTree(lh_coords)

    _, lh_to_rh = rh_tree.query(lh_mirrored)
    _, rh_to_lh = lh_tree.query(rh_mirrored)

    mirror_map = np.empty(n_lh + n_rh, dtype=int)
    mirror_map[:n_lh] = lh_to_rh + n_lh
    mirror_map[n_lh:] = rh_to_lh

    return mirror_map


def setup_inverse(dataset, subject, raw_baseline, ad_hoc_resting=False):
    save_file = Path("./inverse") / dataset / subject / "operator.fif"
    save_file.parent.mkdir(parents=True, exist_ok=True)
    
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)

    src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
    bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    
    if save_file.exists():
        return read_inverse_operator(save_file.absolute()), src, bem

    f = mne.make_forward_solution(raw_baseline.info, trans='fsaverage', src=src, bem=bem, eeg=True)

    noise_cov = mne.compute_raw_covariance(
        raw_baseline, 
        method='shrunk'
    ) if not ad_hoc_resting else mne.make_ad_hoc_cov(raw_baseline.info)

    inverse_operator = make_inverse_operator(
        raw_baseline.info, 
        forward=f, 
        noise_cov=noise_cov, 
        loose="auto",
        depth=0.8
    )
    
    write_inverse_operator(save_file, inverse_operator, overwrite=True)
    
    return inverse_operator, src, bem 

def compute_gain(prepared_inv: InverseOperator, raw: mne.io.Raw | mne.io.RawArray,
                 inverse_mirror_map: NDArray[np.intp], lambda2: float, timestep: float,
                 include_vol: bool = False, include_csd: bool = False, include_inverse: bool = False,
                 include_raw: bool = False, include_abs: bool = False,
                 include_pos: bool = False, include_neg: bool = False,
                 use_epochs: bool = True) -> ColonyMap:
    colonies = {}
    sfreq = raw.info["sfreq"]

    grouped_annotations = {}
    
    if use_epochs:
        for annotation in raw.annotations:
            group = annotation["description"].item().strip()
            if group not in grouped_annotations:
                grouped_annotations[group] = []
            grouped_annotations[group].append((annotation["onset"].item(), annotation["onset"].item() + annotation["duration"].item()))
    else:
        grouped_annotations[""] = [(0, raw.duration)]

    event_id = {name: i + 1 for i, name in enumerate(grouped_annotations)}

    csd_data = mne.preprocessing.compute_current_source_density(raw.copy()).get_data()
    vol_data = raw.get_data()

    ds = []
    if include_vol:
        ds.append(("vol", vol_data))
    if include_csd:
        ds.append(("csd", csd_data))
    for source, data in ds:
        for group, anns in grouped_annotations.items():
            if (source, group) not in colonies:
                colonies[(source, group)] = Colony(data.shape[0], include_raw=include_raw, include_abs=include_abs, include_pos=include_pos, include_neg=include_neg)

            colony = colonies[(source, group)]

            for start_time, end_time in anns:
                sample = data[:, int(start_time * sfreq):int(end_time * sfreq)]

                if sample.shape[1] < timestep * sfreq:
                    continue

                colony.feed(sample, step=int(timestep * sfreq), sfreq=sfreq)

    if not include_inverse:
        return colonies

    max_dur = max(et - st for anns in grouped_annotations.values() for st, et in anns)
    ann_list = []
    event_rows = []
    
    for group, anns in grouped_annotations.items():
        code = event_id[group]
        for st, et in anns:
            ann_list.append((group, et - st))
            event_rows.append([int(st * sfreq), 0, code])
    
    sorted_pairs = sorted(zip(event_rows, ann_list), key=lambda p: p[0][0])
    event_rows, ann_list = zip(*sorted_pairs)
    events = np.array(event_rows)

    epochs = mne.Epochs(raw, events, event_id=event_id,
        tmin=0, tmax=max_dur, baseline=None, preload=True, verbose=False)
    ann_list = [ann_list[i] for i in epochs.selection]
    stc_gen = apply_inverse_epochs(epochs, prepared_inv,
        lambda2=lambda2,
        method="dSPM", prepared=True,
        return_generator=True)

    for stc, (group, dur) in zip(stc_gen, ann_list):
        actual_samples = min(int(dur * sfreq), stc.data.shape[1])
        sample = stc.data[:, :actual_samples]

        if sample.shape[1] < timestep * sfreq:
            continue

        if ("inverse", group) not in colonies:
            colonies[("inverse", group)] = Colony(sample.shape[0], include_raw=include_raw, include_abs=include_abs, include_pos=include_pos, include_neg=include_neg)

        colony = colonies[("inverse", group)]

        colony.feed(sample, step=int(timestep * sfreq), sfreq=sfreq)
        colony.feed(sample[inverse_mirror_map, :], step=int(timestep * sfreq), sfreq=sfreq)
    
    return colonies

if __name__ == "__main__":
    with Live(Panel("Initializing...", expand=False), auto_refresh=True) as live:
        target_datasets = ["eegmmidb"]
        for dataset_index, dataset in enumerate(target_datasets):
            spec = get_dataset_spec(dataset)
            target_subjects = spec["subjects"][:10]
            for subject_index, subject in enumerate(target_subjects):
                subject_baseline_file, subject_active_files = load_subject(dataset, subject)
                raw_baseline, events_baseline = read_subject_record(dataset, subject_baseline_file)
                
                output_dir = Path("./colonies") / dataset / subject
                output_dir.mkdir(parents=True, exist_ok=True)
                
                inv, src, bem = setup_inverse(dataset, subject, raw_baseline)
                snr = 3.0
                lambda2 = 1.0 / (snr ** 2)
                prepared_inv = prepare_inverse_operator(
                    inv,
                    nave=1,
                    lambda2=lambda2
                )

                src_data = mne.read_source_spaces(src)
                inverse_mirror_map = build_hemisphere_mirror_map(src_data, inv['src'][0]['vertno'], inv['src'][1]['vertno'])

                colonies = {}

                include_raw = False
                include_abs = False
                include_pos = True
                include_neg = True

                target_records = subject_active_files[:20]
                for record_index, record in enumerate(target_records):
                    raw_active, _ = read_subject_record(dataset, record)
                    raw_active.annotations.description = np.array([spec["event_ids"][int(a["description"].item().strip())] for a in raw_active.annotations])

                    sfreq = raw_active.info["sfreq"]

                    band_index = 0
                    for band_name, band in BANDS.items():
                        band_index += 1
                        low = band["low"]
                        high = min(band["high"], sfreq / 2.0 - 1)

                        updated_text = f"""Filtering + processing ({low} Hz - {high} Hz)
Band: {band_name} ({band_index}/{len(BANDS)})
Dataset: {dataset} ({dataset_index + 1}/{len(target_datasets)})
Record: {record_index + 1}/{len(target_records)}
Subject: {subject} ({subject_index + 1}/{len(target_subjects)})"""
                        live.update(Panel(updated_text, title="Colony Processing", expand=False))

                        raw_filtered = raw_active.copy()
                        raw_filtered.filter(l_freq=low, h_freq=high, fir_design='firwin', n_jobs=4)

                        new_colonies = compute_gain(prepared_inv, raw_filtered,
                                                    inverse_mirror_map, lambda2, TIMESTEP,
                                                    include_vol=True, include_csd=True,
                                                    include_inverse=True,
                                                    include_raw=include_raw, include_abs=include_abs,
                                                    include_pos=include_pos, include_neg=include_neg)

                        for (source, group), new_colony in new_colonies.items():
                            colonies[(source, band_name, group)] = new_colony

                lh_vertno = inv['src'][0]['vertno']
                rh_vertno = inv['src'][1]['vertno']
                lh_coordinates = src_data[0]['rr'][lh_vertno]
                rh_coordinates = src_data[1]['rr'][rh_vertno]

                all_xyz_coordinates = np.vstack([lh_coordinates, rh_coordinates])
                
                def write_colony(colony, calc, source, band_name, group):
                    target_dir = output_dir / calc / source / band_name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    with open(target_dir / f"{group}.csv", "w") as f:
                        if source == "inverse":
                                f.write("x,y,z,value\n")
                                for i in range(len(all_xyz_coordinates)):
                                    x, y, z = all_xyz_coordinates[i]
                                    value = colony[i]
                                    f.write(f"{x},{y},{z},{value}\n")
                        else:
                            f.write("electrode,value\n")
                            for i in range(colony.shape[0]):
                                value = colony[i]
                                f.write(f"{raw_baseline.ch_names[i]},{value}\n")
                    
                for (source, band_name, group), colony in colonies.items():
                    if include_raw:
                        write_colony(colony.colony_raw, "raw", source, band_name, group)
                    if include_abs:
                        write_colony(colony.colony_abs, "abs", source, band_name, group)
                    if include_pos:
                        write_colony(colony.colony_pos, "pos", source, band_name, group)
                    if include_neg:
                        write_colony(colony.colony_neg, "neg", source, band_name, group)

# + we need to set this up as a reusable funciton that is agnostic to input stream (raw vs csd vs inverse)
# + we need to set this up to run on all of the records and beyond
    # we also need to aggregate same event across all subjects (this is a per head model thing) 
# + we need to store the forward model for fast reload; but does it change across subjects or datasets?
# + we need to see if strongest vertices across bands are the same or not
    # / they do, in fact, have a bit of non-overlap, depending on the band
    # if not, that increases our feature space, kinda; will have to test with a decoder 
    # where we include vertex data at certain frequency band; this allow duplicate
    # vertices at different bands
# + also need to experiment with the timestep, deviation resolution a bit
    # - what about snr and lambda2?
# - should we test out sLORETTA or does that not work with our setup?


# btw: inverse is kinda easiest for cross-dataset decoding
    # but we should do both cross-dataset (inverse only) and same-dataset (raw, inverse, and csd)
# we gotta find papers that implement raw, csd, or inverse decoders and see if this selection method improves things
    # can weight the input data directory per channel (based on the 99th percentile) if linear-ish model
        # nn.Parameter(weights) with grad vs no_grad? claude mentioned this
    # can retrain with training data weighted perhaps? 
# The comparison to Fisher-score selection or anatomical selection on the same decoders is a result
    # any others I should know about for testing?
# consider: this is a binary encoder for 1 singular type of event, though it can be expended to multi-event
    # as in: give epochs of multiple events, so you know what electrodes are generally shared and active
# any good classifier with this should give a probability for the given event to occur
    # we can do a multi-class probability decoder with this via combination of multiple binary classifiers, one per event

# WE OUGHT TO DO some literature review on electrode/vertex/feature selection methods
# also: let's just find the 32, 64, 128 datasets we want to use so we're not limited to MI
    # + refactor _read_csv_record to be eegmmidb-specific (remove the glob in the spec and put it here)
# + ADD: we need to have a check for a null raw baseline and use the... default noise covariance?
# how do we choose when to mirror and when not to mirror during colony growth?
# looks like we have an unsupervised clusterer on our hands: test by collecting colony on some arbitrary event (or relaxation) 
    # and then do a comparison with CSD & Inverse on the arbitrary epoch and the coalesced colony
        # (we need to test all Vol/CSD/Inverse combinations) to get the match percentage/probability
            # would also need to implement the weighting of the coalesced colony someway
            # also: do we mirror the input epoch? I guess so; you could test with/without
        # oh, we should do this per-band too, to see if the band has an effect on the accuracy
    # cause: inverse handles spatial densities; CSD handles dipoles and provides another form of localization
    # for now, I say we should cut out the top 5% or 10% (for ex, occipital overloading) and see if accuracy goes up
# HEYO: we can integrate both positives and negatives; we weight them accordingly such that a highly weighted pos vertex adds to the probability a lot if the value is positive (and level of positivity can increase certainty relative to the weight perhaps), do the same for negatives
# after that, we can test with a supervised decoder like before; use top nth percentile barrier and see what happens

# also also, we should probably do the mirroring for csd and raw too

# what about the path idea by the way?
    # we can do a 2nd pass using the top n% vertices and then go through every window and get the average activations temporally
# also also, when we have more events, we should cut out the broadband noise that appears in every single one to see if accuracy can improve without cutting the truly valuable electrodes


# anywho: a second report is on the amount of vertex overlap (top 5, 15, 25, 50 vertices) per-event within and across subjects and bands
    # this only applies to the abs and pos colonies
    # we can show that the raw deviation colonies skew *left* and are not very useful
        # claude: apparently this means that most of the cortex is desynchronized (ERD) 
        # and focal sync (ERS) takes over... I don't buy, will have to research
    # "a Jaccard matrix across bands, one number per pair, immediately readable"
        # what?

# null result #1: there is no discernible differene in vertex firings
    # already proven false
# null result #2: feature selection in this way has no significant effect on accuracy


# methods notes: 
    # ICA to kill random shit
    # we're sticking with dMSB https://sdgsreview.org/LifestyleJournal/article/view/7511
    # using (a-b)/b because occipital lobe was fucking things up with constant, relatively small changes, so now gain is relative to the baseline itself
        # this does mean a baseline close to zero can cause a spike, mitigated by averaging and sample size
    # hemispheric mirroring to deal with general lateralization
        # for anything that is heavily lateralized to one side this hurts btw (aka, no generalizability here)
    # using different bands because some events respond better on different bands (again, no generalizability)

# for rigor: report skewness on our colony files
    # claude says skewness coefficient is effect size and fine on its own (since n is large)
    # should go ahead and confirm this
# - The tail ratio: what fraction of total variance/power is carried by the top 10%, 25%, 50% of vertices?
    # / variance is noise
# Gini coefficient: measures concentration. 0 = perfectly uniform (every vertex equal), 1 = all value in one vertex.
# get % of the total gain across all electrodes/vertices is handled by per percentile

# we can include some of the coalesced colonies
    # for ex: (for alpha) show that the inverses of the hand grip events are massively red in the frontal cortex while the CSD show contralateralization
    # for ex: (for alpha) show that the inverses of the imagined hand grip events are massively red in the frontal cortex while the CSD show ipsilateralization
        # we should probably confirm this with, you guessed it, a literature review
        
        

# what about generalization?
    # we know that the occipital lobe takes the cake a lot, so via state machine or something, determine that occipital needs to be removed... somehow
    # also: do you reckon we can calculate per-lobe density? for state machining (somehow)
        # alternatively, we could just kill a lobe we don't want for a certain event (ad hoc choice... should be customizable)
    # and: how do we manage lateralization? we're still flatly doing mirroring everywhere...