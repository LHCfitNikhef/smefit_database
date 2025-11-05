import json
import re
from pathlib import Path

keys_to_ignore = ["best_sm", "scales", "theory_cov"]


def load_data_summary_allowed_orders(summary_path: Path):
    text = summary_path.read_text(encoding="utf-8")

    # try YAML first
    try:
        import yaml

        data = yaml.safe_load(text)
        mapping = {}
        if isinstance(data, dict):
            for section, items in data.items():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if isinstance(it, dict) and "name" in it:
                        name = str(it["name"])
                        ao = it.get("allowed_orders")
                        if ao is None:
                            ao = []
                        mapping[name] = [str(x) for x in ao]
        return mapping
    except Exception:
        # fallback: regex scan for inline entries like: name: NAME, ... allowed_orders: [A, B]
        mapping = {}
        for m in re.finditer(
            r"name:\s*([A-Za-z0-9_\-]+)[^\n\r]*allowed_orders:\s*\[([^\]]*)\]", text
        ):
            name = m.group(1)
            ao_raw = m.group(2)
            ao = [x.strip() for x in ao_raw.split(",") if x.strip()]
            mapping[name] = ao
        return mapping


def find_order_keys_in_theory_json(json_path: Path):
    # load top-level keys that are not in the keys_to_ignore list
    data = json.loads(json_path.read_text(encoding="utf-8"))
    orders = [k for k in data.keys() if k not in keys_to_ignore]
    return orders


def test_theory_orders_are_allowed():
    repo_root = Path(__file__).resolve().parents[1]
    summary = repo_root / "data_summary.yaml"
    assert summary.exists(), f"data_summary.yaml not found at {summary}"

    mapping = load_data_summary_allowed_orders(summary)

    theory_dir = repo_root / "theory"
    assert theory_dir.exists(), f"theory directory not found at {theory_dir}"

    failures = []

    for jf in sorted(theory_dir.rglob("*.json")):
        stem = jf.stem
        orders = find_order_keys_in_theory_json(jf)
        if not orders:
            continue

        if stem not in mapping:
            failures.append(
                f"No data_summary entry for theory file '{stem}' (path: {jf})"
            )
            continue

        allowed = mapping.get(stem, [])
        for o in orders:
            if o not in allowed:
                failures.append(
                    f"Order '{o}' in theory/{jf.name} not listed in allowed_orders for '{stem}': {allowed}"
                )

    assert not failures, "\n" + "\n".join(failures)
