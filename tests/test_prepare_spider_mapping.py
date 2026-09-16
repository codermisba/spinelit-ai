"""SPIDER disc-label mapping tests.

SPIDER numbers disc masks top-down: the highest label in a patient is
L5/S1, the next lower label is L4/L5, and so on.  The radiological IVD
label column uses the same convention (1-based).  These tests lock in the
dynamic per-patient mapping (labels 201-205 for lumbar-only cases, 201-206
when T12/L1 is also present).
"""

import numpy as np
import pandas as pd

from prepare_spider import (
    _build_disc_label_to_level,
    _build_ivd_label_to_level,
)


def _volume_with_labels(labels):
    vol = np.zeros((20, 10, 5), dtype=np.int16)
    for i, lb in enumerate(labels):
        vol[1 + i * 3, :, 1] = lb
    return vol


def test_six_level_mask_maps_to_five_lumbar_discs():
    vol = _volume_with_labels([201, 202, 203, 204, 205, 206])
    mapping = _build_disc_label_to_level(vol)
    assert mapping == {
        202: "L1/L2", 203: "L2/L3", 204: "L3/L4",
        205: "L4/L5", 206: "L5/S1",
    }
    assert 201 not in mapping   # T12/L1 dropped


def test_five_level_mask_is_lumbar_only():
    vol = _volume_with_labels([201, 202, 203, 204, 205])
    mapping = _build_disc_label_to_level(vol)
    assert mapping == {
        201: "L1/L2", 202: "L2/L3", 203: "L3/L4",
        204: "L4/L5", 205: "L5/S1",
    }


def test_six_level_ivd_labelling():
    gradings = pd.DataFrame({"Patient": ["42"] * 6, "IVD label": [1, 2, 3, 4, 5, 6]})
    mapping = _build_ivd_label_to_level(gradings, "42")
    assert mapping == {
        2: "L1/L2", 3: "L2/L3", 4: "L3/L4",
        5: "L4/L5", 6: "L5/S1",
    }
    assert 1 not in mapping   # T12/L1 not a graded lumbar level


def test_five_level_ivd_labelling():
    gradings = pd.DataFrame({"Patient": ["7"] * 5, "IVD label": [1, 2, 3, 4, 5]})
    mapping = _build_ivd_label_to_level(gradings, "7")
    assert mapping == {
        1: "L1/L2", 2: "L2/L3", 3: "L3/L4",
        4: "L4/L5", 5: "L5/S1",
    }


def test_no_disc_labels_returns_empty():
    vol = np.zeros((20, 10, 5), dtype=np.int16)
    vol[:, :, 1] = 1      # vertebrae only, no disc labels
    assert _build_disc_label_to_level(vol) == {}

    gradings = pd.DataFrame({"Patient": ["9"], "IVD label": [1]})
    assert _build_ivd_label_to_level(gradings, "8") == {}   # patient not present