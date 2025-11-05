import yaml
from pathlib import Path
import pytest

COMMONDATA_DIRS = ["commondata", "commondata_projections_L0"]


def load_summary_names(summary_path: Path):
    text = summary_path.read_text(encoding="utf-8")

    names = set()

    data = yaml.safe_load(text)
    if isinstance(data, dict):
        for section, items in data.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and "name" in it:
                    names.add(str(it["name"]))
                elif isinstance(it, str):
                    names.add(it)

    return names


def collect_commondata_file_stems(dir_path: Path):
    stems = set()
    p = Path(dir_path)
    if not p.exists():
        return stems
    for f in p.rglob("*.yaml"):
        stems.add(f.stem)
    return stems


@pytest.mark.parametrize(
    "folder",
    COMMONDATA_DIRS,
)
def test_commondata_folder_present_in_data_summary(folder):
    repo_root = Path(__file__).resolve().parents[1]
    summary = repo_root / "data_summary.yaml"
    assert summary.exists(), f"data_summary.yaml not found at {summary}"

    summary_names = load_summary_names(summary)

    cd = repo_root / folder

    file_stems = collect_commondata_file_stems(cd)

    missing = sorted([s for s in file_stems if s not in summary_names])

    assert not missing, (
        f"Found {len(missing)} files under '{folder}' not present in data_summary.yaml: "
        + ", ".join(missing)
    )
