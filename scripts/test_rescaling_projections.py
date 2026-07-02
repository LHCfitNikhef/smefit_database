"""Check that FCCee projection scenarios rescale statistical errors correctly.

For each descoped/upscoped scenario, statistical errors are expected to scale
as 1/sqrt(lumi_factor) relative to the full-FCC baseline
(commondata_projections_L0), since statistical uncertainties scale with
luminosity as 1/sqrt(L).

Exception: in the "top_only" scenario, only FCCee_365 datasets are upscoped;
every other dataset is expected to be identical to the baseline (lumi_factor == 1).

Usage:
    python scripts/test_rescaling_projections.py [--scenario NAME] [--rtol RTOL]

Exits with status 1 if any dataset fails its expected scaling.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMONDATA_SCENARIOS_DIR = REPO_ROOT / "commondata_projection_FCCee_scenarios"
BASELINE_DIR = REPO_ROOT / "commondata_projections_L0"

# Expected ratio of scenario luminosity to full-FCC baseline luminosity.
EXPECTED_LUMI_FACTOR = {
    "commondata_projections_fccee_30MW_4IP_60lumi": 0.6,
    "commondata_projections_fccee_30MW_2IP_36p5lumi": 0.365,
    "commondata_projection_fccee_upscoped_150lumi_top_only": 1.5,
    "commondata_projection_fccee_upscoped_120lumi_global": 1.2,
}

# Scenarios where only FCCee_365 datasets are rescaled; everything else
# should match the baseline exactly.
TOP_ONLY_SCENARIOS = {"commondata_projection_fccee_upscoped_150lumi_top_only"}


def is_rescaled(scenario: str, dataset_name: str) -> bool:
    """Whether `dataset_name` is expected to be rescaled in `scenario`."""
    if scenario in TOP_ONLY_SCENARIOS:
        return "365" in dataset_name
    return True


def expected_stats(
    baseline_stats: np.ndarray, scenario: str, dataset_name: str
) -> np.ndarray:
    if is_rescaled(scenario, dataset_name):
        return baseline_stats / np.sqrt(EXPECTED_LUMI_FACTOR[scenario])
    return baseline_stats


def check_scenario(scenario: str, rtol: float) -> tuple[list[str], list[str]]:
    """Return (flagged, missing_baseline) dataset names for `scenario`."""
    flagged = []
    missing_baseline = []

    scenario_dir = COMMONDATA_SCENARIOS_DIR / scenario
    for projection in sorted(scenario_dir.iterdir()):
        if not projection.name.startswith("FCCee"):
            continue

        baseline_path = BASELINE_DIR / projection.name
        if not baseline_path.exists():
            missing_baseline.append(projection.name)
            continue

        stats_scenario = np.array(
            yaml.safe_load(projection.read_text())["statistical_error"]
        )
        stats_baseline = np.array(
            yaml.safe_load(baseline_path.read_text())["statistical_error"]
        )

        expected = expected_stats(stats_baseline, scenario, projection.name)
        if not np.allclose(stats_scenario, expected, rtol=rtol):
            flagged.append(projection.name)

    return flagged, missing_baseline


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(EXPECTED_LUMI_FACTOR),
        help="Only check this scenario (default: check all scenarios)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for the scaling comparison (default: 1e-3)",
    )
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else sorted(EXPECTED_LUMI_FACTOR)

    any_issues = False
    for scenario in scenarios:
        flagged, missing_baseline = check_scenario(scenario, args.rtol)
        if not flagged and not missing_baseline:
            print(f"{scenario}: OK")
            continue

        any_issues = True
        print(scenario)
        for name in flagged:
            print(f"  FAILED SCALING  {name}")
        for name in missing_baseline:
            print(f"  MISSING BASELINE  {name}")

    return 1 if any_issues else 0


if __name__ == "__main__":
    sys.exit(main())
