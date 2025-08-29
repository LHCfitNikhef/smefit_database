import json
import yaml
from pathlib import Path
import pytest
import re

SKIP_TOP_LEVEL_KEYS = {"best_sm", "scales", "theory_cov"}

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
THEORY_DIR = REPO_ROOT / "theory"
OPERATORS_FILE = REPO_ROOT / "operators_implemented.yaml"

# Load allowed operators from YAML
with OPERATORS_FILE.open(encoding="utf-8") as f:
    ALLOWED_OPERATORS = set(yaml.safe_load(f)["operators"])


JSON_DIRS = [
    REPO_ROOT / "theory",
    REPO_ROOT / "external_chi2/drell_yan/theory",
]

# Collect all JSON files in the specified directories
JSON_FILES = sorted(f for d in JSON_DIRS for f in d.glob("*.json"))


def _split_factors(s: str):
    return [p.strip() for p in re.split(r"\s*\*\s*", s)] if "*" in s else [s.strip()]


@pytest.mark.parametrize("json_path", JSON_FILES, ids=[p.name for p in JSON_FILES])
def test_SM_is_there(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{json_path.name}: top-level JSON must be an object"

    errors = []  # collect all problems for this file

    for top_key, section in data.items():
        if top_key in SKIP_TOP_LEVEL_KEYS:
            continue
        if not isinstance(section, dict):
            errors.append(
                f"section '{top_key}' should be an object, found {type(section).__name__}"
            )
            continue

        # check that the SM key is in the section
        if "SM" not in section:
            errors.append(f"{json_path.name} → '{top_key}': missing 'SM' key")

    if errors:
        msg = ["SM key violations:"]
        msg.extend(f"- {e}" for e in errors)
        pytest.fail("\n".join(msg))


@pytest.mark.parametrize("json_path", JSON_FILES, ids=[p.name for p in JSON_FILES])
def test_theory_operator_keys_allowed(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{json_path.name}: top-level JSON must be an object"

    errors = []  # collect all problems for this file

    for top_key, section in data.items():
        if top_key in SKIP_TOP_LEVEL_KEYS:
            continue
        if not isinstance(section, dict):
            errors.append(
                f"section '{top_key}' should be an object, found {type(section).__name__}"
            )
            continue

        for contrib_key in section.keys():
            # skip SM contribution
            if contrib_key == "SM":
                continue
            factors = _split_factors(contrib_key)
            unknown = [op for op in factors if op not in ALLOWED_OPERATORS]
            if unknown:
                errors.append(
                    f"{json_path.name} → '{top_key}': "
                    f"contribution '{contrib_key}' → not allowed {unknown}"
                )

    if errors:
        msg = ["Operator allowed list violations:"]
        msg.extend(f"- {e}" for e in errors)
        pytest.fail("\n".join(msg))


@pytest.mark.parametrize("json_path", JSON_FILES, ids=[p.name for p in JSON_FILES])
def test_diag_quadratics_exist(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    errors = []  # collect all problems for this file

    for top_key, section in data.items():
        if top_key in SKIP_TOP_LEVEL_KEYS:
            continue
        if not isinstance(section, dict):
            errors.append(
                f"section '{top_key}' should be an object, found {type(section).__name__}"
            )
            continue

        # check that in the section keys quadratics exist, i.e. that at least one key has *
        # if it does not, then continue, nothing to check as the table is only linear
        if not any("*" in k for k in section.keys()):
            continue

        # Now we gather all the linear terms and avoid SM
        linear = [k for k in section.keys() if "*" not in k and k != "SM"]
        # Now we check that all the diag quadratic exist in the keys
        for l in linear:
            if f"{l}*{l}" not in section:
                errors.append(
                    f"{json_path.name} → '{top_key}': "
                    f"Missing quadratic term '{l}*{l}'"
                )

    if errors:
        msg = ["Quadratic term violations:"]
        msg.extend(f"- {e}" for e in errors)
        pytest.fail("\n".join(msg))


@pytest.mark.parametrize("json_path", JSON_FILES, ids=[p.name for p in JSON_FILES])
def test_offdiag_quadratics_no_duplicate(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    errors = []  # collect all problems for this file

    for top_key, section in data.items():
        if top_key in SKIP_TOP_LEVEL_KEYS:
            continue
        if not isinstance(section, dict):
            errors.append(
                f"section '{top_key}' should be an object, found {type(section).__name__}"
            )
            continue

        # check that in the section keys quadratics exist, i.e. that at least one key has *
        # if it does not, then continue, nothing to check as the table is only linear
        if not any("*" in k for k in section.keys()):
            continue

        # Now we gather all the linear terms and avoid SM
        linear = [k for k in section.keys() if "*" not in k and k != "SM"]
        # Now generate all the possible unique pairs
        pairs = [(l1, l2) for i, l1 in enumerate(linear) for l2 in linear[i + 1 :]]
        # Now we check that all the off-diag quadratic exist in the keys and that they are unique
        for l1, l2 in pairs:
            if f"{l1}*{l2}" in section and f"{l2}*{l1}" in section:
                errors.append(
                    f"{json_path.name} → '{top_key}': "
                    f"Both off-diagonal quadratic terms '{l1}*{l2}' and '{l2}*{l1}' exist"
                )

    if errors:
        msg = ["Off-diagonal quadratic term duplicates:"]
        msg.extend(f"- {e}" for e in errors)
        pytest.fail("\n".join(msg))


@pytest.mark.parametrize("json_path", JSON_FILES, ids=[p.name for p in JSON_FILES])
def test_offdiag_quadratics_exist(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    errors = []  # collect all problems for this file

    for top_key, section in data.items():
        if top_key in SKIP_TOP_LEVEL_KEYS:
            continue
        if not isinstance(section, dict):
            errors.append(
                f"section '{top_key}' should be an object, found {type(section).__name__}"
            )
            continue

        # check that in the section keys quadratics exist, i.e. that at least one key has *
        # if it does not, then continue, nothing to check as the table is only linear
        if not any("*" in k for k in section.keys()):
            continue

        # Now we gather all the linear terms and avoid SM
        linear = [k for k in section.keys() if "*" not in k and k != "SM"]
        # Now generate all the possible unique pairs
        pairs = [(l1, l2) for i, l1 in enumerate(linear) for l2 in linear[i + 1 :]]
        # Now we check that all the off-diag quadratic exist in the keys and that they are unique
        for l1, l2 in pairs:
            if f"{l1}*{l2}" not in section and f"{l2}*{l1}" not in section:
                errors.append(
                    f"{json_path.name} → '{top_key}': "
                    f"Missing off-diagonal quadratic term '{l1}*{l2}'"
                )

    if errors:
        msg = ["Off-diagonal quadratic term violations:"]
        msg.extend(f"- {e}" for e in errors)
        pytest.fail("\n".join(msg))
