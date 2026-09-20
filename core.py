from pathlib import Path
from typing import Literal

Band = Literal["whole", "standard", "theta", "delta", "alpha", "beta", "gamma"]

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

DATASET_SPECS = {
    "grasplift": {
        "root": Path("./datasets/grasplift/train"),
        "sfreq": 500.0,
        "subjects": [f"subj{i}" for i in range(1, 12 + 1)],
        "channels": [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "FC5", "FC1", "FC2", "FC6",
            "T7", "C3", "Cz", "C4", "T8",
            "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
            "P7", "P3", "Pz", "P4", "P8",
            "PO9", "O1", "Oz", "O2", "PO10",
        ],
        "event_ids": {
            1: "HandStart",
            2: "FirstDigitTouch",
            3: "BothStartLoadPhase",
            4: "LiftOff",
            5: "Replace",
            6: "BothReleased",
        },
        "event_time_padding": (-100 / 1000, -100 / 1000), # s
        "ignore_events": [3, 6],
    },
}

def get_dataset_spec(dataset: str):
    try:
        return DATASET_SPECS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc