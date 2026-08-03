"""Master script: build a new globally-rescaled FCC-ee projection scenario.

Runs, in order:
  1. rescale_descoped_fccee.py       -- rescales FCCee_* statistical errors
  2. rescale_hadr_systematics.py     -- rescales H_HADR systematics
  3. rescale_zdata_systematics.py    -- rescales Zdata systematics
  4. symlink_non_fcc_projections.sh  -- symlinks in the non-FCC datasets

Step 1 is parametrized by `lumi_scale` and writes into the scenario
directory derived from it. Steps 2-3 accept the new scenario directory
via `--target-dir`/`--lumi-scale`, which they rescale on top of the
entries already hardcoded in their own `TARGET_LUMI_SCALE` tables (see
their docstrings) -- no need to pre-register the new scenario there.
Step 4 accepts the new directory as its documented target-dir argument
too, appending it to its own hardcoded scenario list for this run.

Usage:
    python scripts/build_fccee_global_scenario.py <lumi_scale>

lumi_scale: ratio of the new scenario's global FCC-ee luminosity to
            nominal (e.g. 1.2 for a 120% "upscoped" scenario). The
            output directory name follows the existing convention,
            e.g. 1.2 -> commondata_projection_fccee_upscoped_120lumi_global.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCENARIOS_DIR = REPO_ROOT / "commondata_projection_FCCee_scenarios"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _format_lumi(lumi_scale: float) -> str:
    """1.2 -> '120', 0.365 -> '36p5' (matches existing scenario-folder naming)."""
    value = lumi_scale * 100
    if value == int(value):
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _scenario_dir(lumi_scale: float) -> Path:
    name = f"commondata_projection_fccee_upscoped_{_format_lumi(lumi_scale)}lumi_global"
    return SCENARIOS_DIR / name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "lumi_scale",
        type=float,
        help="Global FCC-ee luminosity scale factor relative to nominal (e.g. 1.2)",
    )
    args = parser.parse_args()
    if args.lumi_scale <= 0:
        parser.error("lumi_scale must be a positive number")

    output_dir = _scenario_dir(args.lumi_scale)

    print(f"[1/4] Rescaling FCC-ee statistical errors -> {output_dir}")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "rescale_descoped_fccee.py"),
            str(args.lumi_scale),
            str(output_dir),
        ],
        check=True,
    )

    print("\n[2/4] Rescaling H_HADR systematics")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "rescale_hadr_systematics.py"),
            "--target-dir",
            str(output_dir),
            "--lumi-scale",
            str(args.lumi_scale),
        ],
        check=True,
    )

    print("\n[3/4] Rescaling Zdata systematics")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "rescale_zdata_systematics.py"),
            "--target-dir",
            str(output_dir),
            "--lumi-scale",
            str(args.lumi_scale),
        ],
        check=True,
    )

    print("\n[4/4] Symlinking non-FCC projections")
    subprocess.run(
        ["bash", str(SCRIPTS_DIR / "symlink_non_fcc_projections.sh"), str(output_dir)],
        check=True,
        cwd=SCRIPTS_DIR,
    )

    print(f"\nDone. Scenario written to {output_dir}")


if __name__ == "__main__":
    main()
