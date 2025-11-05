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


def test_external_chi2_files_and_classes_listed_in_yaml():
    """Scan `external_chi2/` for python files and their top-level classes, and
    verify that each (file, class) pair is recorded in
    `ext_likelihood_summary.yaml` (i.e. there's a YAML key equal to the class
    name whose resolved path points to that file).
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    yaml_path = os.path.join(repo_root, "ext_likelihood_summary.yaml")
    assert os.path.exists(yaml_path), f"Could not find {yaml_path}"

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    external = data.get("external_chi2", {}) if isinstance(data, dict) else {}
    # Build mapping: resolved absolute path -> set of YAML keys that point to it
    path_to_keys = {}
    for key, info in external.items():
        path_str = info.get("path") if isinstance(info, dict) else None
        resolved = _resolve_path(repo_root, path_str)
        if not resolved:
            continue
        abs_resolved = os.path.abspath(resolved)
        path_to_keys.setdefault(abs_resolved, set()).add(key)

    # Walk external_chi2 directory for .py files
    ext_dir = os.path.join(repo_root, "external_chi2")
    assert os.path.isdir(ext_dir), f"external_chi2 directory not found at {ext_dir}"

    missing = []
    for root, dirs, files in os.walk(ext_dir):
        # skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            file_path = os.path.abspath(os.path.join(root, fname))
            try:
                src = open(file_path, "r", encoding="utf-8").read()
            except Exception as e:
                missing.append((file_path, None, f"could-not-read: {e}"))
                continue
            try:
                tree = ast.parse(src, filename=file_path)
            except SyntaxError as e:
                missing.append((file_path, None, f"syntax-error: {e}"))
                continue

            # collect top-level class names (ignore private starting with _)
            classes = [
                n.name
                for n in tree.body
                if isinstance(n, ast.ClassDef) and not n.name.startswith("_")
            ]

            for cls in classes:
                keys_for_file = path_to_keys.get(file_path, set())
                if cls not in keys_for_file:
                    missing.append((file_path, cls, "not-listed-in-yaml"))

    if missing:
        msgs = [f"{fp}: class={cls} -> {reason}" for (fp, cls, reason) in missing]
        pytest.fail(
            "Some external_chi2 files/classes are not listed in ext_likelihood_summary.yaml:\n"
            + "\n".join(msgs)
        )
