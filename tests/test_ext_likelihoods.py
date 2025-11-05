import os
import ast
import pytest

try:
    import yaml
except Exception:  # pragma: no cover - skip if yaml missing
    yaml = None


def _resolve_path(repo_root, path_str):
    """Resolve a path from the YAML into an actual path in the repo.

    If the YAML contains an absolute path that includes 'external_chi2', we
    take the part after that and join with repo_root/external_chi2. If not,
    we try interpreting the path as relative to the repo root.
    """
    if not isinstance(path_str, str):
        return None
    marker = "external_chi2"
    if marker in path_str:
        # take everything after the first occurrence of external_chi2
        sub = path_str.split(marker, 1)[1].lstrip("/\\")
        candidate = os.path.normpath(os.path.join(repo_root, marker, sub))
        return candidate
    # fallback: join relative to repo root
    return os.path.normpath(os.path.join(repo_root, path_str))


def test_external_chi2_classes_exist():
    """Ensure for every entry in `external_chi2` the target file defines a
    class with the same name as the YAML key.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    yaml_path = os.path.join(repo_root, "ext_likelihood_summary.yaml")
    assert os.path.exists(yaml_path), f"Could not find {yaml_path}"

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    external = data.get("external_chi2", {}) if isinstance(data, dict) else {}
    assert external, "No 'external_chi2' section found in ext_likelihood_summary.yaml"

    missing_classes = []
    for name, info in external.items():
        path_str = None
        if isinstance(info, dict):
            path_str = info.get("path")
        resolved = _resolve_path(repo_root, path_str)

        if not resolved or not os.path.exists(resolved):
            # let the path-existence test handle this; report here for clarity
            missing_classes.append((name, resolved, "file-not-found"))
            continue

        try:
            src = open(resolved, "r", encoding="utf-8").read()
        except Exception as e:
            missing_classes.append((name, resolved, f"could-not-read: {e}"))
            continue

        try:
            tree = ast.parse(src, filename=resolved)
        except SyntaxError as e:
            missing_classes.append((name, resolved, f"syntax-error: {e}"))
            continue

        found = any(
            isinstance(node, ast.ClassDef) and node.name == name
            for node in ast.walk(tree)
        )
        if not found:
            missing_classes.append((name, resolved, "class-not-found"))

    if missing_classes:
        msgs = [f"{n}: {r} -> {reason}" for (n, r, reason) in missing_classes]
        pytest.fail("External chi2 class presence failures:\n" + "\n".join(msgs))
