#!/usr/bin/env python3
"""
Update README.md with Wilson-coefficient definitions parsed from operators_implemented.yaml.

- Reads:  operators_implemented.yaml  (mapping 'operators': {OpName: "expr", ...})
- Writes: README.md (replaces content between markers):
    <!-- BEGIN: SMEFT-OPERATORS -->
    ... generated content ...
    <!-- END: SMEFT-OPERATORS -->
"""

from __future__ import annotations
from pathlib import Path
import re
import yaml
from collections import defaultdict
from pathlib import Path

# Get repo root (one level up from this script's directory)
ROOT = Path(__file__).resolve().parent.parent

IN_YAML = ROOT / "operators_implemented.yaml"
README = ROOT / "README.md"

BEGIN = "<!-- BEGIN: SMEFT-OPERATORS -->"
END = "<!-- END: SMEFT-OPERATORS -->"

KNOWN_CATEGORIES = {
    "Bosonic",
    "Dipoles",
    "Quark currents",
    "Yukawa",
    "4Heavy four-quarks",
    "4Heavy (right-handed bottom, only included in tttt and ttbb datasets)",
    "2L2H four-quarks",
    "Lepton currents",
    "Four-lepton",
    "2L2Q operators",
    "2L2q operators",
}


def op_to_coeff_name(op: str) -> str:
    return ("c" + op[1:]) if op.startswith("O") else f"c{op}"


def extract_categories_by_op(yaml_text: str) -> dict[str, str]:
    """
    Recover category comments (# Category) inside the 'operators:' block.
    """
    lines = yaml_text.splitlines()
    categories_by_op: dict[str, str] = {}
    in_ops = False
    current_category = "Uncategorised"
    key_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*\".*\"")
    for line in lines:
        if not in_ops and re.match(r"^\s*operators\s*:\s*$", line):
            in_ops = True
            continue
        if not in_ops:
            continue
        # If we hit a new top-level mapping key, bail
        if re.match(r"^[^\s#-]", line):
            break
        mcat = re.match(r"^\s*#\s*(.+?)\s*$", line)
        if mcat:
            candidate = mcat.group(1).strip()
            if candidate in KNOWN_CATEGORIES:
                current_category = candidate
            continue
        mkey = key_re.match(line)
        if mkey:
            categories_by_op[mkey.group(1)] = current_category
    return categories_by_op


def build_generated_section(
    ops_map: dict[str, str], cat_by_op: dict[str, str], raw_yaml: str
) -> str:
    # Track category order as they appear in YAML
    seen_cats = []
    for op in ops_map.keys():
        c = cat_by_op.get(op, "Uncategorised")
        if c not in seen_cats:
            seen_cats.append(c)

    # Group by category preserving insertion order
    grouped = defaultdict(list)
    for op, expr in ops_map.items():
        grouped[cat_by_op.get(op, "Uncategorised")].append((op_to_coeff_name(op), expr))

    header = (
        "## SMEFiT Wilson coefficients\n\n"
        "Definitions are given in terms of the "
        "[Warsaw basis (WCxf)](https://wcxf.github.io/assets/pdf/SMEFT.Warsaw.pdf).\n\n"
        "This section is auto-generated from `operators_implemented.yaml`. "
        "Do not edit it manually.\n\n"
        "Each entry defines the Wilson coefficient `cX`, corresponding to the SMEFiT operator `OX`, "
        "as used in the JSON data tables.\n\n"
    )
    parts = [header]
    for cat in seen_cats:
        parts.append(f"### {cat}\n\n")
        parts.append("```text\n")
        for coeff, expr in grouped[cat]:
            parts.append(f"{coeff} = {expr}\n")
        parts.append("```\n\n")
    return "".join(parts)


def main():
    if not IN_YAML.exists():
        raise SystemExit(f"Missing {IN_YAML}")

    yaml_text = IN_YAML.read_text(encoding="utf-8")
    data = yaml.safe_load(yaml_text)
    ops = data.get("operators")
    if not isinstance(ops, dict):
        raise SystemExit("YAML must contain a top-level mapping 'operators'")

    cat_by_op = extract_categories_by_op(yaml_text)
    generated = build_generated_section(ops, cat_by_op, yaml_text)

    readme_text = README.read_text(encoding="utf-8") if README.exists() else ""
    begin_idx = readme_text.find(BEGIN)
    end_idx = readme_text.find(END)

    if begin_idx == -1 or end_idx == -1 or end_idx < begin_idx:
        # Append section with markers at the end
        new_text = readme_text.rstrip() + "\n\n" + BEGIN + "\n" + generated + END + "\n"
    else:
        # Replace content between markers (exclusive)
        before = readme_text[: begin_idx + len(BEGIN)] + "\n"
        after = "\n" + readme_text[end_idx:]
        new_text = before + generated + after

    README.write_text(new_text, encoding="utf-8")
    print(f"Updated {README} between markers.")


if __name__ == "__main__":
    main()
