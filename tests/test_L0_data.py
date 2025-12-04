import os
import json
from glob import glob
import pathlib
from pathlib import Path
import pytest


import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
THEORY_DIR = REPO_ROOT / "theory"

BASE_DIR = os.path.dirname(__file__)
PROJECTION_DIR = [
    os.path.join(BASE_DIR, "..", "commondata_projections_L0"),
    os.path.join(BASE_DIR, "..", "external_chi2/drell_yan/commondata_projections_L0"),
]

JSON_FILES = [pathlib.Path(f) for f in glob(os.path.join(THEORY_DIR, "*.json"))]


def get_yaml_files(directories):
    """Return all .yaml files in the directory."""
    files = []
    for directory in directories:
        if os.path.exists(directory):
            files.extend(
                pathlib.Path(os.path.join(directory, f))
                for f in os.listdir(directory)
                if f.endswith(".yaml") and not f.endswith("uncor.yaml")
            )
    return files


@pytest.mark.parametrize("yaml_file", get_yaml_files(PROJECTION_DIR))
def test_dataset_name_matches_filename(yaml_file):
    """Check that the dataset_name in the YAML matches the filename."""

    filename = yaml_file.stem
    theory_file = yaml_file.parent / "../theory" / f"{filename}.json"

    # we never have projections that end with "uncor"

    with open(theory_file) as f:
        theory_data = json.load(f)

    with open(yaml_file) as f:
        projection_L0 = yaml.safe_load(f)

    best_sm = theory_data.get("best_sm")
    if len(best_sm) == 1:
        best_sm = best_sm[0]

    data_central = projection_L0.get("data_central")

    assert np.allclose(
        best_sm, data_central, atol=0.0
    ), f"L0 projection does not match best_sm for {filename}"
