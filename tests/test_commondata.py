import os
import yaml
import pytest

# Path to your commondata folder
BASE_DIR = os.path.dirname(__file__)
COMMONDATA_DIRS = [
    os.path.join(BASE_DIR, "..", "commondata"),
    os.path.join(BASE_DIR, "..", "commondata_projections_L0"),
    os.path.join(BASE_DIR, "..", "commondata_projections_L1"),
]


def get_yaml_files(directories):
    """Return all .yaml files in the directory."""
    files = []
    for directory in directories:
        if os.path.exists(directory):
            files.extend(
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.endswith(".yaml")
            )
    return files


@pytest.mark.parametrize("yaml_file", get_yaml_files(COMMONDATA_DIRS))
def test_dataset_name_matches_filename(yaml_file):
    """Check that the dataset_name in the YAML matches the filename."""
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    filename_without_ext = os.path.splitext(os.path.basename(yaml_file))[0]
    dataset_name = data.get("dataset_name")

    assert (
        dataset_name == filename_without_ext
    ), f"dataset_name '{dataset_name}' does not match filename '{filename_without_ext}'"
