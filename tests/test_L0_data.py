import os
import json
from glob import glob
import pathlib
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
THEORY_DIR = REPO_ROOT / "theory"
PROJECTION_DIR = REPO_ROOT / "commondata_projections_L0"

JSON_FILES = [pathlib.Path(f) for f in glob(os.path.join(THEORY_DIR, "*.json"))]

for theory_path in JSON_FILES:
    filename = theory_path.stem
    projection_L0_path = PROJECTION_DIR / f"{filename}.yaml"

    # we never have projections that end with "uncor"
    if filename.endswith("uncor"):
        continue

    if not os.path.exists(projection_L0_path):
        continue

    with open(theory_path) as f:
        theory_data = json.load(f)
    with open(projection_L0_path) as f:
        projection_L0 = yaml.safe_load(f)

    best_sm = theory_data.get("best_sm")
    if len(best_sm) == 1:
        best_sm = best_sm[0]

    data_central = projection_L0.get("data_central")
    assert np.allclose(
        best_sm, data_central, rtol=1e-3
    ), f"L0 projection does not match best_sm for {filename}"
