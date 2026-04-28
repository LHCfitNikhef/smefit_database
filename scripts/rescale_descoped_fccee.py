"""Rescale statistical uncertainties in descoped FCCee commondata files.

Delegates projection to `smefit PROJ`, calling it once per energy group with
the appropriate absolute luminosity. Values are always derived from the
original FCCee_* files, so the script is safe to run repeatedly.

Optim-obs datasets (no luminosity field) are skipped — their covariance lives
in the external .dat files and must be rescaled separately.

Usage:
    python scripts/rescale_descoped_fccee.py <lumi_scale> [--runcard PATH]

lumi_scale: ratio of descoped to nominal FCC-ee luminosity per energy group.
            E.g. 0.3 means 30% of nominal luminosity for each group.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parent.parent
COMMONDATA_DIR = REPO_ROOT / "commondata_projections_L0"
THEORY_DIR = REPO_ROOT / "theory"
DEFAULT_RUNCARD = (
    REPO_ROOT / "runcards/projections/descoped_FCCee/descoped_fccee_4ips.yaml"
)

# Nominal luminosities (fb^-1) per FCCee energy group.
# FCCee_365 is intentionally omitted — not part of the descoped run plan.
NOMINAL_LUMI = {
    "FCCee_91": 205000,
    "FCCee_161": 19200,
    "FCCee_240": 10800,
}


class _YamlfmtDumper(yaml.Dumper):
    """PyYAML dumper that indents list items to match yamlfmt's style."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow=flow, indentless=False)


def _yaml_dump(data, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(
            data,
            f,
            Dumper=_YamlfmtDumper,
            default_flow_style=False,
            sort_keys=False,
            width=10000,
        )


def get_datasets_per_group() -> dict[str, list[dict]]:
    """Read dataset names and theory orders per group from data_summary.yaml,
    skipping optim_obs datasets (those without a luminosity field)."""
    summary = yaml.safe_load((REPO_ROOT / "data_summary.yaml").read_text())
    groups = {}
    for group in NOMINAL_LUMI:
        datasets = []
        for entry in summary.get(group, []):
            cd = yaml.safe_load((COMMONDATA_DIR / f"{entry['name']}.yaml").read_text())
            if "luminosity" not in cd:
                print(
                    f"  SKIPPED {entry['name']} (no luminosity — external covariance)"
                )
                continue
            datasets.append({"name": entry["name"], "order": entry["order"]})
        if datasets:
            groups[group] = datasets
    return groups


def rescale(lumi_scale: float, runcard_template: Path) -> None:
    template = yaml.safe_load(runcard_template.read_text())
    groups = get_datasets_per_group()

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)

        for group, datasets in groups.items():
            nominal_lumi = NOMINAL_LUMI[group]
            target_lumi = lumi_scale * nominal_lumi
            proj_dir = tmp / group
            proj_dir.mkdir()

            runcard = dict(template)
            runcard["commondata_path"] = str(COMMONDATA_DIR)
            runcard["theory_path"] = str(THEORY_DIR)
            runcard["projections_path"] = str(proj_dir)
            runcard["datasets"] = datasets

            rc_path = tmp / f"{group}.yaml"
            _yaml_dump(runcard, rc_path)

            print(
                f"\n[{group}] nominal={nominal_lumi} → target={target_lumi:.1f}"
                f"  ({len(datasets)} datasets)"
            )
            subprocess.run(
                ["smefit", "PROJ", "--lumi", str(target_lumi), str(rc_path)],
                check=True,
            )

            for out_file in sorted(proj_dir.glob("FCCee_*.yaml")):
                data = yaml.safe_load(out_file.read_text())
                data["dataset_name"] = f"descoped_{data['dataset_name']}"
                dest = COMMONDATA_DIR / f"descoped_{out_file.name}"
                _yaml_dump(data, dest)
                print(f"  Written: {dest.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "lumi_scale",
        type=float,
        help="Luminosity scaling factor (descoped / nominal FCC-ee per group)",
    )
    parser.add_argument(
        "--runcard",
        type=Path,
        default=DEFAULT_RUNCARD,
        help="Runcard template (default: descoped_fccee_4ips.yaml)",
    )
    args = parser.parse_args()

    if args.lumi_scale <= 0:
        parser.error("lumi_scale must be a positive number")

    rescale(args.lumi_scale, args.runcard)


if __name__ == "__main__":
    main()
