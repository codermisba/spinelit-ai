"""
build_ddd_labels.py
===================

Build ``dataset/ddd_labels.csv`` from the SPIDER radiology gradings **without**
re-downloading the 3.7 GB ``images.zip``.

``prepare_spider.py`` derives the same file while it processes the image
volumes. When the processed JPGs already exist locally (they do — see
``dataset/data/processed_spider_jpgs``), only the small ``radiological_gradings.csv``
is needed, so this script regenerates the labels directly.

Mapping convention (identical to ``prepare_spider.py``)
-------------------------------------------------------
The radiology CSV ``IVD label`` column is 1-based, top-down. The **highest**
IVD label for a patient is L5/S1; lower numbers move up the lumbar spine.
Labels above L1/L2 (e.g. T12/L1) are dropped, giving the pipeline's
``[L1/L2, L2/L3, L3/L4, L4/L5, L5/S1]`` order.

Output columns
--------------
    filename, level, pfirrmann_grade

Usage
-----
    python build_ddd_labels.py
    python build_ddd_labels.py --gradings dataset/data/spider_raw/radiological_gradings.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import DISC_LEVELS, DDD_LABELS_CSV, ROOT_DIR

PFRRMANN_COL = "Pfirrman grade"   # exact column name in the SPIDER CSV
DEFAULT_GRADINGS = (ROOT_DIR / "dataset" / "data" / "spider_raw"
                    / "radiological_gradings.csv")
COORDS_CSV = ROOT_DIR / "dataset" / "coords_pretrain.csv"


def ivd_label_to_level(gradings: pd.DataFrame, patient_id: str) -> dict[int, str]:
    """Map a patient's 1-based IVD labels to lumbar disc levels."""
    pat = gradings[gradings["Patient"].astype(str) == str(patient_id)]
    if pat.empty:
        return {}
    labels = sorted(pat["IVD label"].dropna().astype(int).unique())
    if not labels:
        return {}
    max_label = labels[-1]          # highest IVD label = L5/S1
    mapping: dict[int, str] = {}
    for label in labels:
        idx = max_label - label     # 0 = L5/S1 (last in DISC_LEVELS)
        if 0 <= idx < len(DISC_LEVELS):
            mapping[label] = DISC_LEVELS[-(idx + 1)]
    return mapping


def build(gradings_csv: Path, coords_csv: Path, output: Path) -> pd.DataFrame:
    gradings = pd.read_csv(gradings_csv)
    if "Patient" not in gradings.columns:
        raise SystemExit(f"'Patient' column missing in {gradings_csv}")
    if PFRRMANN_COL not in gradings.columns:
        raise SystemExit(f"'{PFRRMANN_COL}' column missing in {gradings_csv}")

    # Restrict to the spider cases that actually have landmark/disc annotations.
    if coords_csv.exists():
        coords = pd.read_csv(coords_csv)
        spider = coords[coords["source"] == "spider"]["filename"].unique()
        available = {str(name) for name in spider}
    else:
        available = None

    rows: list[dict] = []
    for patient_id, pat in gradings.groupby("Patient"):
        mapping = ivd_label_to_level(gradings, patient_id)
        if not mapping:
            continue
        # A patient may have multiple scan types; apply the grade to every
        # processed spider filename belonging to that patient.
        if available is not None:
            case_names = [n for n in available
                          if n.split("_")[0] == str(patient_id)]
        else:
            case_names = [f"{patient_id}_t2.jpg"]
        for label, level in mapping.items():
            subset = pat[pat["IVD label"].astype(str) == str(label)]
            if subset.empty:
                continue
            value = subset.iloc[0][PFRRMANN_COL]
            try:
                grade = int(float(str(value).strip()))
            except (TypeError, ValueError):
                continue
            if not 1 <= grade <= 5:
                continue
            for name in case_names:
                rows.append({"filename": name, "level": level,
                             "pfirrmann_grade": grade})

    frame = pd.DataFrame(rows, columns=["filename", "level",
                                        "pfirrmann_grade"])
    frame = frame.drop_duplicates(subset=["filename", "level"], keep="first")
    frame = frame.sort_values(["filename", "level"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradings", default=str(DEFAULT_GRADINGS))
    parser.add_argument("--coords", default=str(COORDS_CSV))
    parser.add_argument("--output", default=str(DDD_LABELS_CSV))
    args = parser.parse_args()

    frame = build(Path(args.gradings), Path(args.coords), Path(args.output))
    print(f"wrote {args.output}")
    print(f"label rows     : {len(frame)}")
    print(f"cases with DDD : {frame['filename'].nunique()}")
    print("per-level rows :")
    print(frame["level"].value_counts().reindex(DISC_LEVELS).to_string())
    print("grade counts   :")
    print(frame["pfirrmann_grade"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
