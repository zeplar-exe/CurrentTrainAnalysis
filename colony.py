from analyze import get_subject_data, RECORDS
from mne.minimum_norm import apply_inverse_raw, write_inverse_operator, read_inverse_operator, make_inverse_operator
import mne
import numpy as np
import scipy
from pathlib import Path

class Colony:
    DEVIATION_RESOLUTION = 5.0 # s
    
    def __init__(self, size: int, include_raw: bool = True, include_abs: bool = True, include_pos: bool = True):
        self.include_raw = include_raw
        self.include_abs = include_abs
        self.include_pos = include_pos
        self.colony_raw = np.zeros(size)
        self.colony_abs = np.zeros(size)
        self.colony_pos = np.zeros(size)
        
    def feed(self, data: np.ndarray, step: int):
        a = np.abs(scipy.signal.hilbert(data))
        b = scipy.ndimage.uniform_filter1d(a, size=int(Colony.DEVIATION_RESOLUTION * sfreq), axis=1)
        data = a-b
            
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
            last_matrix = current_matrix

BANDS = {
    "whole": {
        "low": 1,
        "high": 100,
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

EVENT_IDS = {
    "physionet": {
        1: "T0",
        2: "T1",
        3: "T2"
    }
}

TIMESTEP = 50 / 1000 # s
ASSUMED_EVENT_LENGTH = 800 / 1000 # s

def load_subject(dataset, subject):
    if dataset == "physionet":
        r = []
        
        for record in RECORDS:
            subject = record[5:9]
            session = record[9:12]
            
            if subject == subject:
                r.append(Path("./physionet-motorimagery") / subject / f"{subject}{session}.edf")
        
        if not r:
            raise ValueError(f"Subject {subject} not found in dataset {dataset}")
        
        record_baseline = r[0]
        
        return record_baseline, r[2:]
    
def fix_raw(dataset, raw):
    raw.notch_filter(freqs=60.0, fir_design='firwin')
    
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

def read_subject_record(dataset, record):
    raw = mne.io.read_raw_edf(record, preload=True)
    fix_raw(dataset, raw)
    events, _ = mne.events_from_annotations(raw)
    
    return raw, events
      
def setup_inverse(dataset, subject, raw_baseline):
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
    )

    inverse_operator = make_inverse_operator(
        raw_baseline.info, 
        forward=f, 
        noise_cov=noise_cov, 
        loose="auto",
        depth=0.8
    )
    
    write_inverse_operator(save_file, inverse_operator, overwrite=True)
    
    return inverse_operator, src, bem
    

for subject in ["S001", "S002", "S003"]:
    print("Processing inverse and colonies for subject:", subject)
    subject_baseline_file, subject_active_files = load_subject("physionet", subject)
    raw_baseline, events_baseline = read_subject_record("physionet", subject_baseline_file)
    
    output_dir = Path("./colonies") / "physionet" / subject
    output_dir.mkdir(parents=True, exist_ok=True)
    
    inv, src, bem = setup_inverse("physionet", subject, raw_baseline)
            
    colonies = {}
    
    lh_vertno = 0
    rh_vertno = 0

    for record in subject_active_files[:3]:
        raw_active, events_active = read_subject_record("physionet", record)
    
        snr = 3.0
        stc = apply_inverse_raw(
            raw_active, 
            inv, 
            lambda2=1.0 / (snr ** 2),
            buffer_size=5000,
            method="dSPM", 
            pick_ori=None
        )
        
        lh_vertno = stc.lh_vertno
        rh_vertno = stc.rh_vertno

        sfreq = raw_active.info["sfreq"]
        grouped_annotations = {}

        annotations_active = mne.annotations_from_events(
            events=events_active, 
            sfreq=raw_active.info['sfreq']
        )

        for annotation in annotations_active:
            desc = int(annotation["description"].item().strip())
            key = EVENT_IDS["physionet"][desc]
            
            if key not in grouped_annotations:
                grouped_annotations[key] = []
            grouped_annotations[key].append((annotation["onset"].item(), annotation["onset"].item() + ASSUMED_EVENT_LENGTH))

        print(f"Grouped Annotations: {grouped_annotations}")

        for band_name, band in BANDS.items():
            print(f"Processing colony on band: {band_name}")
            low = band["low"]
            high = min(band["high"], sfreq / 2.0 - 1)
            vertex_data = mne.filter.filter_data(
                stc.copy().data, 
                sfreq=sfreq, 
                l_freq=low, 
                h_freq=high, 
                fir_design='firwin'
            )
            
            raw_data = raw_active.copy()
            raw_data.filter(l_freq=low, h_freq=high, fir_design='firwin')
            raw_data = raw_data.get_data()
            
            csd_data = mne.preprocessing.compute_current_source_density(raw_active.copy())
            csd_data.filter(l_freq=low, h_freq=high, fir_design='firwin')
            csd_data = csd_data.get_data()
            
            for (source, data) in {"raw": raw_data, "csd": csd_data, "inverse": vertex_data}.items():
                for group, annotations in grouped_annotations.items():
                    if group not in colonies:
                        colonies[(source, band_name, group)] = Colony(data.shape[0], include_raw=True, include_abs=True, include_pos=True)
                    print(f"Processing colony for group: {group}")
                    colony = colonies[(source, band_name, group)]
                    
                    for annotation in annotations:
                        start_sample = int(annotation[0] * sfreq)
                        end_sample = int(annotation[1] * sfreq)
                        colony.feed(data[:, start_sample:end_sample], step=int(TIMESTEP * sfreq))
    
    src_data = mne.read_source_spaces(src)

    lh_coordinates = src_data[0]['rr'][lh_vertno]
    rh_coordinates = src_data[1]['rr'][rh_vertno]

    all_xyz_coordinates = np.vstack([lh_coordinates, rh_coordinates])
        
    for (source, band_name, group), colony in colonies.items():
        event_name = EVENT_IDS["physionet"][group]
        with open(output_dir / f"colony_raw_{event_name}_{band_name}.csv", "w") as f:
            f.write("x,y,z,value\n")
            for i in range(len(all_xyz_coordinates)):
                x, y, z = all_xyz_coordinates[i]
                value = colony.colony_raw[i]
                f.write(f"{x},{y},{z},{value}\n")
        with open(output_dir / f"colony_abs_{event_name}_{band_name}.csv", "w") as f:
            f.write("x,y,z,value\n")
            for i in range(len(all_xyz_coordinates)):
                x, y, z = all_xyz_coordinates[i]
                value = colony.colony_abs[i]
                f.write(f"{x},{y},{z},{value}\n")
        with open(output_dir / f"colony_pos_{event_name}_{band_name}.csv", "w") as f:
            f.write("x,y,z,value\n")
            for i in range(len(all_xyz_coordinates)):
                x, y, z = all_xyz_coordinates[i]
                value = colony.colony_pos[i]
                f.write(f"{x},{y},{z},{value}\n")

# WE OUGHT TO DO some literature review on electrode/vertex/feature selection methods

# we need to set this up as a reusable funciton that is agnostic to input stream (raw vs csd vs inverse)
# we need to set this up to run on all of the records and beyond
    # we also need to aggregate same event across all subjects (this is a per-dataset thing) 
# we need to store the forward model for fast reload; but does it change across subjects or datasets?
# we need to see if strongest vertices across bands are the same or not
    # if not, that increases our feature space, kinda; will have to test with a decoder 
    # where we include vertex data at certain frequency band; this allow duplicate
    # vertices at different bands
# also need to experiment with the timestep, event length, deviation resolution a bit
    # waht about snr and lambda2
    # should we test out sLORETTA or does that not work with our setup?

# and I guess for choosing which colonies to keep
    # mean should be greater than median (right skewed)
    # skewness should be at least medium (at least 0.4?)
# for rigor: report skewness on our colony files
    # claude says skewness coefficient is effect size and fine on its own (since n is large)
    # should go ahead and confirm this

# anywho: a second report is on the amount of vertex overlap (top 5, 10, 25, 50 vertices) per-event within and across subjects
    # this only applies to the abs and pos colonies
    # we can show that the raw deviation colonies skew *left* and are not very useful
        # claude: apparently this means that most of the cortex is desynchronized (ERD) 
        # and focal sync (ERS) takes over... I don't buy, will have to research
    # "a Jaccard matrix across bands, one number per pair, immediately readable"
        # what?

# The tail ratio: what fraction of total variance/power is carried by the top 10%, 25%, 50% of vertices?
# Gini coefficient: measures concentration. 0 = perfectly uniform (every vertex equal), 1 = all value in one vertex.

# btw: inverse is kinda easiest for cross-dataset decoding
    # but we should do both cross-dataset (inverse only) and same-dataset (raw, inverse, and csd)
# we gotta find papers that implement raw, csd, or inverse decoders and see if this selection method improves things
    # can weight the input data directory per channel (based on the 99th percentile) if linear-ish model
        # nn.Parameter(weights) with grad vs no_grad? claude mentioned this
    # can retrain with training data weighted perhaps? 
# The comparison to Fisher-score selection or anatomical selection on the same decoders is a result
    # any others I should know about for testing?

# null result #1: there is no discernible differene in vertex firings
    # already proven false
# null result #2: feature selection in this way has no significant effect on accuracy

# what about the path idea by the way?
    # we can do a 2nd pass using the top n% vertices and then go through every window and get the average activations temporally
# also also, when we have more events, we should cut out the broadband noise that appears in every single one