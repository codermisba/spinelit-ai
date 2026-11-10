"""
Tests for the real-data preparation helpers.

These lock the two conventions that make the DDD and segmentation heads honest:

* SPIDER IVD label 1..N maps top-down, with the highest label = L5/S1.
* SPIDER mask legend maps to 0 background / 1 vertebra / 2 disc / 3 canal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from build_ddd_labels import build, ivd_label_to_level
from prepare_segmentation_pairs import map_classes


def _gradings():
    rows = []
    for patient, labels in (("7", [1, 2, 3, 4, 5, 6]),
                            ("8", [3, 4, 5, 6, 7])):
        for i, label in enumerate(labels):
            rows.append({
                "Patient": patient,
                "IVD label": label,
                "Pfirrman grade": (i % 5) + 1,
            })
    return pd.DataFrame(rows)


def test_ivd_labels_map_from_the_bottom():
    gradings = _gradings()
    # Patient 7 has six IVDs (T12/L1..L5/S1); the top one is dropped.
    mapping = ivd_label_to_level(gradings, "7")
    assert mapping[2] == "L1/L2"
    assert mapping[3] == "L2/L3"
    assert mapping[4] == "L3/L4"
    assert mapping[5] == "L4/L5"
    assert mapping[6] == "L5/S1"
    assert 1 not in mapping

    # Patient 8 has five IVDs; label 3 is L1/L2, label 7 is L5/S1.
    mapping8 = ivd_label_to_level(gradings, "8")
    assert mapping8[3] == "L1/L2"
    assert mapping8[7] == "L5/S1"


def test_build_ddd_labels_emits_one_row_per_case_level(tmp_path):
    gradings = _gradings()
    gradings_csv = tmp_path / "gradings.csv"
    gradings.to_csv(gradings_csv, index=False)

    coords = pd.DataFrame({
        "filename": ["7_t2.jpg", "8_t2.jpg"],
        "source": ["spider", "spider"],
        "level": ["L1/L2", "L1/L2"],
        "relative_x": [0.5, 0.5],
        "relative_y": [0.5, 0.5],
    })
    coords_csv = tmp_path / "coords.csv"
    coords.to_csv(coords_csv, index=False)

    out = tmp_path / "ddd_labels.csv"
    frame = build(gradings_csv, coords_csv, out)

    assert set(frame.columns) == {"filename", "level", "pfirrmann_grade"}
    assert frame["filename"].nunique() == 2
    assert len(frame) == 10                     # 2 cases x 5 lumbar discs
    assert not frame.duplicated(["filename", "level"]).any()
    assert out.exists()


def test_mask_legend_maps_to_pipeline_classes():
    raw = np.zeros((4, 6), dtype=np.int16)
    raw[0, 0] = 1        # vertebra
    raw[0, 1] = 25       # vertebra
    raw[0, 2] = 101      # partial vertebra
    raw[0, 3] = 125      # partial vertebra
    raw[1, 0] = 201      # disc
    raw[1, 1] = 225      # disc
    raw[1, 2] = 100      # spinal canal
    raw[2, 0] = 50       # unknown label -> background

    mapped = map_classes(raw)
    assert mapped.dtype == np.uint8
    assert mapped[0, 0] == 1 and mapped[0, 1] == 1
    assert mapped[0, 2] == 1 and mapped[0, 3] == 1
    assert mapped[1, 0] == 2 and mapped[1, 1] == 2
    assert mapped[1, 2] == 3
    assert mapped[2, 0] == 0
    assert mapped[3].tolist() == [0, 0, 0, 0, 0, 0]
