# query by subjects, bands, events, datasets; perhaps cluster after this

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from colony import Colony, ColonySource, ColonyType, MultiColony
from core import Band, DATASET_SPECS, get_dataset_spec

COALESCE_ROOT = Path("coalesce")
COLONIES_ROOT = Path("colonies")


class SubjectSet:
    def __init__(self, dataset: str, subjects: list[str]):
        get_dataset_spec(dataset)
        self.dataset = dataset
        self.subjects = subjects
        
    @staticmethod
    def all(dataset: str):
        s = get_dataset_spec(dataset)
        return SubjectSet(dataset, s["subjects"])

class EventSet:
    def __init__(self, dataset: str, events: list[str]):
        get_dataset_spec(dataset)
        self.dataset = dataset
        self.events = events
    
    @staticmethod
    def all(dataset: str):
        s = get_dataset_spec(dataset)
        return EventSet(dataset, list(s["event_ids"].values()))

class ColonyQuery:
    def __init__(self, subjects: list[SubjectSet], events: list[EventSet], bands: list[Band], sources: list[ColonySource], colony_types: list[ColonyType]):
        self.subjects = subjects
        self.bands = bands
        self.events = events
        self.sources = sources
        self.colony_types = colony_types

class ColonyQueryResult:
    def __init__(self, dataset: str, subject: str, event: str, band: Band, source: ColonySource, typ: ColonyType, colony: Colony):
        self.dataset = dataset
        self.subject = subject
        self.event = event
        self.band = band
        self.source = source
        self.type = typ
        self.colony = colony


def _load_coalesced(dataset: str, source: ColonySource, band: Band, event: str, types: list[ColonyType]) -> Colony | None:
    include_pos = "pos" in types
    include_neg = "neg" in types
    include_raw = "raw" in types
    include_abs = "abs" in types

    pos_path = COALESCE_ROOT / dataset / source / band / event / "pos.csv"
    neg_path = COALESCE_ROOT / dataset / source / band / event / "neg.csv"

    pos_df = pd.read_csv(pos_path) if (include_pos or include_raw or include_abs) and pos_path.exists() else None
    neg_df = pd.read_csv(neg_path) if (include_neg or include_raw or include_abs) and neg_path.exists() else None

    if pos_df is None and neg_df is None:
        return None

    size = len(pos_df) if pos_df is None else len(neg_df)
    colony = Colony(size, include_raw=include_raw, include_abs=include_abs, include_pos=include_pos, include_neg=include_neg)

    if pos_df is not None and include_pos:
        colony.colony_pos = pos_df["value"].values
    if neg_df is not None and include_neg:
        colony.colony_neg = neg_df["value"].values

    return colony


def _load_subject_colony(dataset: str, subject: str, source: ColonySource, band: Band, event: str, types: list[ColonyType], mirrored: bool = False) -> Colony | None:
    mirror_key = "mirrored" if mirrored else "regular"
    subject_dir = COLONIES_ROOT / dataset / subject

    pos_path = subject_dir / "pos" / source / band / mirror_key / f"{event}.csv"
    neg_path = subject_dir / "neg" / source / band / mirror_key / f"{event}.csv"

    include_pos = "pos" in types
    include_neg = "neg" in types
    include_raw = "raw" in types
    include_abs = "abs" in types

    pos_df = pd.read_csv(pos_path) if (include_pos or include_raw or include_abs) and pos_path.exists() else None
    neg_df = pd.read_csv(neg_path) if (include_neg or include_raw or include_abs) and neg_path.exists() else None

    if pos_df is None and neg_df is None:
        return None

    size = len(pos_df) if pos_df is None else len(neg_df)
    colony = Colony(size, include_raw=include_raw, include_abs=include_abs, include_pos=include_pos, include_neg=include_neg)

    if pos_df is not None and include_pos:
        colony.colony_pos = pos_df["value"].values
    if neg_df is not None and include_neg:
        colony.colony_neg = neg_df["value"].values

    return colony


def get_colonies(query: ColonyQuery) -> list[ColonyQueryResult]:
    results = []

    datasets_with_events = defaultdict(list)
    for es in query.events:
        datasets_with_events[es.dataset].extend(es.events)

    datasets_with_subjects = defaultdict(list)
    for ss in query.subjects:
        datasets_with_subjects[ss.dataset].extend(ss.subjects)

    for dataset, events in datasets_with_events.items():
        subjects = datasets_with_subjects.get(dataset)

        for band in query.bands:
            for source in query.sources:
                for event in events:
                    if subjects:
                        for subject in subjects:
                            colony = _load_subject_colony(dataset, subject, source, band, event, query.colony_types)
                            if colony is None:
                                continue
                            for typ in query.colony_types:
                                results.append(ColonyQueryResult(dataset, subject, event, band, source, typ, colony))
                    else:
                        colony = _load_coalesced(dataset, source, band, event, query.colony_types)
                        if colony is None:
                            continue
                        for typ in query.colony_types:
                            results.append(ColonyQueryResult(dataset, "coalesced", event, band, source, typ, colony))

    return results