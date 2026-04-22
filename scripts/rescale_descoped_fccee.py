"""Rescale statistical uncertainties in descoped FCCee commondata files.

Values are always derived from the original FCCee_* files, so the script
is safe to run repeatedly with different factors.

Usage:
    python scripts/rescale_descoped_fccee.py <lumi_scale>

lumi_scale: ratio of descoped luminosity to vanilla FCC-ee luminosity.
            E.g. 0.3 means the descoped run plan has 30% of the nominal luminosity.
            Statistical errors scale as 1/sqrt(lumi_scale).
"""

import argparse
import math
from pathlib import Path

import yaml


COMMONDATA_DIR = Path(__file__).parent.parent / "commondata_projections_L0"


class _YamlfmtDumper(yaml.Dumper):
    """PyYAML dumper that indents list items to match yamlfmt's style."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow=flow, indentless=False)


def rescale(lumi_scale: float) -> None:
    stat_scale = 1.0 / math.sqrt(lumi_scale)
    descoped_files = sorted(COMMONDATA_DIR.glob("descoped_FCCee_*.yaml"))

    if not descoped_files:
        raise FileNotFoundError(
            f"No descoped_FCCee_*.yaml files found in {COMMONDATA_DIR}"
        )

    for descoped_path in descoped_files:
        original_name = descoped_path.name.replace("descoped_FCCee_", "FCCee_", 1)
        original_path = COMMONDATA_DIR / original_name

        if not original_path.exists():
            print(f"WARNING: original file not found, skipping: {original_path.name}")
            continue

        with open(original_path) as f:
            data = yaml.safe_load(f)

        if "luminosity" not in data:
            # optim_obs files carry no luminosity — their covariance lives in the
            # external .dat files and must be rescaled separately.
            print(
                f"SKIPPED  {descoped_path.name}  (no luminosity field — external covariance)"
            )
            continue

        data["dataset_name"] = descoped_path.stem
        data["luminosity"] = data["luminosity"] * lumi_scale
        data["statistical_error"] = [v * stat_scale for v in data["statistical_error"]]

        with open(descoped_path, "w") as f:
            yaml.dump(
                data,
                f,
                Dumper=_YamlfmtDumper,
                default_flow_style=False,
                sort_keys=False,
                width=10000,
            )

        print(
            f"Updated {descoped_path.name}  (lumi x{lumi_scale}, stat x{stat_scale:.6f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "lumi_scale",
        type=float,
        help="Luminosity scaling factor (descoped / vanilla FCC-ee)",
    )
    args = parser.parse_args()

    if args.lumi_scale <= 0:
        parser.error("lumi_scale must be a positive number")

    rescale(args.lumi_scale)


if __name__ == "__main__":
    main()
