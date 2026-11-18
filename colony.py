from collections import defaultdict

from mne.minimum_norm import prepare_inverse_operator, apply_inverse_raw, apply_inverse_epochs, write_inverse_operator, read_inverse_operator, make_inverse_operator
import mne
from mne.minimum_norm import InverseOperator
import numpy as np
from numpy.typing import NDArray
import warnings
import scipy
from scipy.spatial import KDTree
from pathlib import Path
from rich.live import Live
from rich.panel import Panel
from typing import Literal
from core import COLONIES_ROOT, DATASET_SPECS, BANDS, get_dataset_spec

warnings.filterwarnings("ignore", message="FastICA did not converge")
warnings.filterwarnings("ignore", message=".*does not conform to MNE naming conventions.*")

ColonySource = Literal["vol", "csd", "inverse"]
ColonyType = Literal["raw", "abs", "pos", "neg"]
ColonyMap = dict[tuple[ColonySource, str], "MultiColony"]

TIMESTEP = 25 / 1000 # s
MULTICOLONY_STEP = 75 / 1000 # s


class Colony:
    DEVIATION_RESOLUTION = 2.0 # s

    def __init__(self, size: int, include_raw: bool = False, include_abs: bool = False, include_pos: bool = False, include_neg: bool = False):
        self.size = size
        self.include_raw = include_raw
        self.include_abs = include_abs
        self.include_pos = include_pos
        self.include_neg = include_neg
        if include_raw:
            self.colony_raw = np.zeros(size)
        if include_abs:
            self.colony_abs = np.zeros(size)
        if include_pos: 
            self.colony_pos = np.zeros(size)
        if include_neg:
            self.colony_neg = np.zeros(size)
        
    @staticmethod
    def _safe_weights(numerator: np.ndarray, denominator: float) -> np.ndarray:
        if np.isclose(denominator, 0.0):
            return np.zeros_like(numerator, dtype=float)
        
        vals = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )
        return np.nan_to_num(np.minimum(1, vals), nan=0.0, posinf=0.0, neginf=0.0)
        
    def feed(self, data: np.ndarray, step: int, sfreq: int, deviation=True):
        if deviation:
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
        pc = np.percentile(self.colony_raw, 99.9)
        return Colony._safe_weights(self.colony_raw, pc)

    def abs_weights(self):
        pc = np.percentile(self.colony_abs, 99.9)
        return Colony._safe_weights(self.colony_abs - np.min(self.colony_abs), pc)

    def pos_weights(self):
        pc = np.percentile(self.colony_pos, 99.9)
        return Colony._safe_weights(self.colony_pos - np.min(self.colony_pos), pc)

    def neg_weights(self):
        pc = np.percentile(np.abs(self.colony_neg), 99.9)
        return Colony._safe_weights(self.colony_neg - np.max(self.colony_neg), -pc)

    def merge(self, other: "Colony"):
        if self.size != other.size:
            raise ValueError(f"Cannot merge Colonies with different sizes: {self.size} vs {other.size}")
        
        if self.include_raw and other.include_raw:
            self.colony_raw += other.colony_raw
        if self.include_abs and other.include_abs:
            self.colony_abs += other.colony_abs
        if self.include_pos and other.include_pos:
            self.colony_pos += other.colony_pos
        if self.include_neg and other.include_neg:
            self.colony_neg += other.colony_neg


class MultiColony:
    def __init__(self, colonies: list[Colony], size: int, interval: float):
        for colony in colonies:
            if colony.size != size:
                raise ValueError(f"Cannot create MultiColony with Colonies of different sizes: {size} vs {colony.size}")
        
        self.colonies = colonies
        self.size = size
        self.interval = interval
    
    def raw_weights(self, cut_zero=True):
        last_idx = len(self.colonies)
        if cut_zero:
            while last_idx > 0 and not np.any(self.colonies[last_idx - 1].colony_raw):
                last_idx -= 1
        return np.array([colony.raw_weights() for colony in self.colonies[:last_idx]])
    
    def abs_weights(self, cut_zero=True):
        last_idx = len(self.colonies)
        if cut_zero:
            while last_idx > 0 and not np.any(self.colonies[last_idx - 1].colony_abs):
                last_idx -= 1
        return np.array([colony.abs_weights() for colony in self.colonies[:last_idx]])
    
    def pos_weights(self, cut_zero=True):
        last_idx = len(self.colonies)
        if cut_zero:
            while last_idx > 0 and not np.any(self.colonies[last_idx - 1].colony_pos):
                last_idx -= 1
        return np.array([colony.pos_weights() for colony in self.colonies[:last_idx]])

    def neg_weights(self, cut_zero=True):
        last_idx = len(self.colonies)
        if cut_zero:
            while last_idx > 0 and not np.any(self.colonies[last_idx - 1].colony_neg):
                last_idx -= 1
        return np.array([colony.neg_weights() for colony in self.colonies[:last_idx]])
    
    def merge(self, other: "MultiColony"):
        if self.interval != other.interval:
            raise ValueError(f"Cannot merge MultiColonies with different intervals: {self.interval} vs {other.interval}")
        if self.size != other.size:
            raise ValueError(f"Cannot merge MultiColonies with different sizes: {self.size} vs {other.size}")
        
        diff = len(other.colonies) - len(self.colonies)
        if diff > 0:
            ref = self.colonies[0]
            self.colonies.extend([Colony(self.size, include_raw=ref.include_raw, include_abs=ref.include_abs, include_pos=ref.include_pos, include_neg=ref.include_neg) for _ in range(diff)])
        for i, colony in enumerate(other.colonies):
            if colony.size != self.size:
                raise ValueError(f"Cannot merge Colonies with different sizes: {self.size} vs {colony.size}")
            self.colonies[i].merge(colony)


def _read_grasplift_record(record: Path, spec: dict):
    signal_path = Path(record)
    events_path = signal_path.with_name(signal_path.name.replace("_data.csv", "_events.csv"))

    n_ch = len(spec["channels"])
    signal = np.loadtxt(signal_path, delimiter=",", skiprows=1, usecols=range(1, n_ch + 1), dtype=float)

    raw = mne.io.RawArray(signal.T * 1e-6, mne.create_info(spec["channels"], spec["sfreq"], ch_types="eeg"), verbose="error")
    raw.set_montage("standard_1005")
    raw.set_eeg_reference(projection=True)

    event_names = list(spec["event_ids"].values())
    name_to_code = {v: k for k, v in spec["event_ids"].items()}
    events_data = np.loadtxt(events_path, delimiter=",", skiprows=1, usecols=range(1, len(event_names) + 1), dtype=int)

    onsets = []
    durations = []
    descriptions = []
    event_rows = []

    ignore = set(spec.get("ignore_events", []))

    for col_idx, event_name in enumerate(event_names):
        code = name_to_code[event_name]
        if code in ignore:
            continue

        col = events_data[:, col_idx]
        diff = np.diff(col, prepend=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if len(ends) < len(starts):
            ends = np.append(ends, len(col))

        for s, e in zip(starts, ends):
            onset_time = s / spec["sfreq"]
            dur = min((e - s) / spec["sfreq"], raw.duration - onset_time)
            onsets.append(onset_time)
            durations.append(dur)
            descriptions.append(str(code))
            event_rows.append([s, 0, code])
        
    onsets = np.array(onsets)
    durations = np.array(durations)
    descriptions = np.array(descriptions)

    padding_back, padding_forward = spec["event_time_padding"] if isinstance(spec["event_time_padding"], tuple) else (0, 0)
    raw.set_annotations(mne.Annotations(onset=onsets + padding_back, duration=durations + padding_forward, description=descriptions))
    events = np.array(event_rows) if event_rows else np.empty((0, 3), dtype=int)

    return raw, events

def load_subject(dataset, subject):
    spec = get_dataset_spec(dataset)

    if dataset == "grasplift":
        series = sorted(spec["root"].glob(f"{subject}_series*_data.csv"),
                        key=lambda p: int(p.stem.split("_series")[1].split("_")[0]))
        if not series:
            raise ValueError(f"Subject {subject} not found in dataset {dataset}")
        return series[0], series[1:]

    raise ValueError(f"Unsupported dataset: {dataset}")
    
def fix_raw(dataset, raw):
    raw.notch_filter(freqs=60.0, fir_design='firwin')
    
    ica = mne.preprocessing.ICA(n_components=0.995, method='fastica')
    ica.fit(raw.copy().filter(1, min(100, raw.info["sfreq"] / 2 - 1), fir_design='firwin'))
    muscle_idx, scores = ica.find_bads_muscle(raw)
    ica.exclude = muscle_idx
    raw = ica.apply(raw.copy())
        
    return raw

def read_subject_record(dataset, record):
    spec = get_dataset_spec(dataset)

    if dataset == "grasplift":
        raw, events = _read_grasplift_record(record, spec)
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

def setup_inverse(dataset, subject, raw_baseline, ad_hoc_resting=False, info=None, root=Path("./inverse")):
    save_file = root / dataset / subject / "operator.fif"
    save_file.parent.mkdir(parents=True, exist_ok=True)
    
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)

    src = fs_dir / "bem" / "fsaverage-ico-3-src.fif"
    bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    
    if save_file.exists():
        return read_inverse_operator(save_file.absolute()), src, bem

    f = mne.make_forward_solution(raw_baseline.info if raw_baseline else info, trans='fsaverage', src=src, bem=bem, eeg=True)

    noise_cov = mne.compute_raw_covariance(
        raw_baseline, 
        method='shrunk'
    ) if not ad_hoc_resting else mne.make_ad_hoc_cov(raw_baseline.info if raw_baseline else info)

    inverse_operator = make_inverse_operator(
        raw_baseline.info if raw_baseline else info,
        forward=f, 
        noise_cov=noise_cov, 
        loose="auto",
        depth=0.8
    )
    
    write_inverse_operator(save_file, inverse_operator, overwrite=True)
    
    return inverse_operator, src, bem 

def compute_gain(prepared_inv: InverseOperator, raw: mne.io.Raw | mne.io.RawArray,
                 lambda2: float, timestep: float, colony_step: float, inverse_mirror_map: NDArray[np.intp] | None = None,
                 include_vol: bool = False, include_csd: bool = False, include_inverse: bool = False,
                 include_raw: bool = False, include_abs: bool = False,
                 include_pos: bool = False, include_neg: bool = False,
                 use_epochs: bool = True) -> ColonyMap:
    if abs((colony_step / timestep) - round(colony_step / timestep)) > 1e-6:
        raise ValueError(f"Colony step ({colony_step}) must be a multiple of timestep ({timestep})")
    
    multis: ColonyMap = {}
    colonies: dict[tuple[ColonySource, str], list[Colony]] = defaultdict(list)
    sfreq = raw.info["sfreq"]

    grouped_annotations: dict[str, list[tuple[float, float]]] = {}

    if use_epochs:
        for annotation in raw.annotations:
            group = annotation["description"].item().strip()
            if group not in grouped_annotations:
                grouped_annotations[group] = []
            grouped_annotations[group].append((annotation["onset"].item(), annotation["onset"].item() + annotation["duration"].item()))
    else:
        grouped_annotations[""] = [(0, raw.times[-1])]
    
    event_id = {name: i + 1 for i, name in enumerate(grouped_annotations)}


    ds = []
    if include_vol:
        vol_data = raw.get_data()
        ds.append(("vol", vol_data))
    if include_csd:
        csd_data = mne.preprocessing.compute_current_source_density(raw.copy()).get_data()
        ds.append(("csd", csd_data))
    for source, data in ds:
        for group, anns in grouped_annotations.items():
            cs = colonies[(source, group)]

            for start_time, end_time in anns:
                dur = end_time - start_time
                t = 0
                i = 0
                while t < dur:
                    t0 = t
                    t += colony_step
                    sample = data[:, int((start_time + t0) * sfreq):int(min(end_time, start_time + t) * sfreq)]
                
                    if sample.shape[1] < timestep * sfreq:
                        sample = np.pad(sample, ((0, 0), (0, int(timestep * sfreq - sample.shape[1]))))
                    
                    if i >= len(cs):
                        colony = Colony(sample.shape[0], include_raw=include_raw, include_abs=include_abs, include_pos=include_pos, include_neg=include_neg)
                        cs.append(colony)
                    else:
                        colony = cs[i]
                        
                    colony.feed(sample, step=int(timestep * sfreq), sfreq=sfreq)
                    i += 1
 
    if include_inverse:
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

        mne_epochs = mne.Epochs(raw, events, event_id=event_id,
            tmin=0, tmax=max_dur, baseline=None, preload=True, event_repeated="merge", verbose=False)
        ann_list = [ann_list[i] for i in mne_epochs.selection]
        stc_gen = apply_inverse_epochs(mne_epochs, prepared_inv,
            lambda2=lambda2,
            method="dSPM", prepared=True,
            return_generator=True,
            verbose=False)

        for stc, (group, dur) in zip(stc_gen, ann_list):
            actual_samples = min(int(dur * sfreq), stc.data.shape[1])
            sample = stc.data[:, :actual_samples]

            cs = colonies[("inverse", group)]
            
            t = 0
            i = 0
            while t < dur:
                t0 = t
                t += colony_step
                window = sample[:, int(t0 * sfreq):int(min(dur, t) * sfreq)]

                if window.shape[1] < timestep * sfreq:
                    window = np.pad(window, ((0, 0), (0, int(timestep * sfreq - window.shape[1]))))

                if i >= len(cs):
                    colony = Colony(window.shape[0], include_raw=include_raw, include_abs=include_abs, include_pos=include_pos, include_neg=include_neg)
                    colony.feed(window, step=int(timestep * sfreq), sfreq=sfreq)
                    cs.append(colony)
                else:
                    cs[i].feed(window, step=int(timestep * sfreq), sfreq=sfreq)
                
                i += 1

    for (source, group), colony_list in colonies.items():
        multis[(source, group)] = MultiColony(colony_list, size=colony_list[0].size, interval=colony_step)

    return multis

if __name__ == "__main__":
    with Live(Panel("Initializing...", expand=False), auto_refresh=True) as live:
        target_datasets = ["grasplift"]
        for dataset_index, dataset in enumerate(target_datasets):
            spec = get_dataset_spec(dataset)
            target_subjects = spec["subjects"]
            for subject_index, subject in enumerate(target_subjects):
                subject_baseline_file, subject_active_files = load_subject(dataset, subject)
                output_dir = COLONIES_ROOT / dataset / subject
                output_dir.mkdir(parents=True, exist_ok=True)
                
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
                inverse_mirror_map = build_hemisphere_mirror_map(src_data, inv['src'][0]['vertno'], inv['src'][1]['vertno'])

                reg_colonies: dict[tuple, MultiColony] = {}
                mirror_colonies: dict[tuple, MultiColony] = {}

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
                        
                        new_regular_colonies = compute_gain(prepared_inv, raw_filtered,
                                                    lambda2, TIMESTEP, MULTICOLONY_STEP, None,
                                                    include_vol=True, include_csd=True, include_inverse=True,
                                                    include_raw=include_raw, include_abs=include_abs,
                                                    include_pos=include_pos, include_neg=include_neg)
                        new_mirred_colonies = compute_gain(prepared_inv, raw_filtered,
                                                    lambda2, TIMESTEP, MULTICOLONY_STEP, inverse_mirror_map,
                                                    include_vol=True, include_csd=True, include_inverse=True,
                                                    include_raw=include_raw, include_abs=include_abs,
                                                    include_pos=include_pos, include_neg=include_neg)

                        for (source, group), new_colony in new_regular_colonies.items():
                            k = (source, band_name, group)
                            if k in reg_colonies:
                                reg_colonies[k].merge(new_colony)
                            else:
                                reg_colonies[k] = new_colony
                        for (source, group), new_colony in new_mirred_colonies.items():
                            k = (source, band_name, group)
                            if k in mirror_colonies:
                                mirror_colonies[k].merge(new_colony)
                            else:
                                mirror_colonies[k] = new_colony

                lh_vertno = inv['src'][0]['vertno']
                rh_vertno = inv['src'][1]['vertno']
                lh_coordinates = src_data[0]['rr'][lh_vertno]
                rh_coordinates = src_data[1]['rr'][rh_vertno]

                all_xyz_coordinates = np.vstack([lh_coordinates, rh_coordinates])
                
                def write_colony(colony: MultiColony, calc, source, band_name, group, mirror_key):
                    if len(colony.colonies) == 0:
                        return
                    
                    target_dir = output_dir / calc / source / band_name / mirror_key
                    target_dir.mkdir(parents=True, exist_ok=True)
                    relevant_arrs = []
                    
                    for colony_instance in colony.colonies:
                        if calc == "raw":
                            relevant_arrs.append(colony_instance.colony_raw)
                        elif calc == "abs":
                            relevant_arrs.append(colony_instance.colony_abs)
                        elif calc == "pos":
                            relevant_arrs.append(colony_instance.colony_pos)
                        elif calc == "neg":
                            relevant_arrs.append(colony_instance.colony_neg)
                    
                    vals = [f"value{i}" for i in range(len(relevant_arrs))]
                    
                    with open(target_dir / f"{group}.csv", "w") as f:
                        if source == "inverse":
                            f.write(f"x,y,z,{",".join(vals)}\n")
                            for i in range(len(all_xyz_coordinates)):
                                x, y, z = all_xyz_coordinates[i]
                                values = [str(a[i]) for a in relevant_arrs]
                                f.write(f"{x},{y},{z},{','.join(values)}\n")
                        else:
                            f.write("electrode,value\n")
                            for i in range(relevant_arrs[0].shape[0]):
                                value = [str(a[i]) for a in relevant_arrs]
                                f.write(f"{raw_baseline.ch_names[i]},{','.join(value)}\n")
                
                for mirror_key, colonies in {"regular": reg_colonies, "mirrored": mirror_colonies}.items():
                    for (source, band_name, group), colony in colonies.items():
                        if include_raw:
                            write_colony(colony, "raw", source, band_name, group, mirror_key)
                        if include_abs:
                            write_colony(colony, "abs", source, band_name, group, mirror_key)
                        if include_pos:
                            write_colony(colony, "pos", source, band_name, group, mirror_key)
                        if include_neg:
                            write_colony(colony, "neg", source, band_name, group, mirror_key)

# WE OUGHT TO DO some literature review on electrode/vertex/feature selection methods
    # + https://link.springer.com/article/10.1186/s13634-015-0251-9#Sec3


# overall: how does this compare to CSP *at the very least*?
# hell, just because an eleectrode is active does not mean that it is discriminative; that seems to be the main result
# 0. need to choose a study or some
    # 0.1. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0361026
# 1. take colonies per-subject, per-band, coalesced over event
# 2. set a cutoff for SD or other variance metric where a subject-band colony is too diffuse and thus a poor discriminator
    # 2.1. or better, use this as a discriminability/weight score per band
# 3. cluster subject coalesce colonies per-band and either have one classifier per cluster, or take each cluster and use weights from each
    # 3.1. alternatively, provide a way to determine what cluster a subject falls under
    # 3.2. we also need to decide on a few percentiles to use; subject clustering uses percentiles
    # 3.3. can we create a across-subject clusterer using numbers instead of overlap?
    # 3.4 anywho: a secondary report is on the amount of vertex overlap per-event within and across subjects and bands
# 4. test existing decoders on weighted clusters (where mathematically applicable) and top-n node limiter, same as pnpl
    # 4.1. we can test with both whole-event colonies and multicolonies; note the temporal distinction
# 5. by the way, we ought to test with all 4 colony modes; perhaps some are better than others for certain BCI tasks
    # 5.1. also notable to mention that phase does not matter when computing colonies
    # 5.2. can weight the input data directory per channel (based on the 99th percentile) if linear-ish model
        # nn.Parameter(weights) with grad vs no_grad? claude mentioned this
    # 5.3. can retrain with training data weighted perhaps? 
    # 5.4. The comparison to Fisher-score selection or anatomical selection on the same decoders is a result
        # any others I should know about for testing?    


# VISUALS
# we can include some of the coalesced colonies
    # for ex: (for alpha) show that the inverses of the hand grip events are massively red in the frontal cortex while the CSD show contralateralization
    # for ex: (for alpha) show that the inverses of the imagined hand grip events are massively red in the frontal cortex while the CSD show ipsilateralization
        # we should probably confirm this with, you guessed it, a literature review
# for rigor: report skewness on our colony files
    # claude says skewness coefficient is effect size and fine on its own (since n is large)
    # should go ahead and confirm this  
    

# methods notes: 
    # ICA to kill random shit
    # we're sticking with dMSB https://sdgsreview.org/LifestyleJournal/article/view/7511
    # using (a-b)/b because occipital lobe was fucking things up with constant, relatively small changes, so now gain is relative to the baseline itself
        # this does mean a baseline close to zero can cause a spike, mitigated by averaging and sample size
    # using different bands because some events respond better on different bands (need a citation)
    # limitation: not dealing with different sized electrode arrays for EEG and CSD