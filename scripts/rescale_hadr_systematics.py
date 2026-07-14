"""Rescale FCCee/LEP3 H_HADR systematic uncertainties in projection scenarios.

`rescale_descoped_fccee.py` (and the `smefit PROJ` machinery it wraps) only
rescales statistical uncertainties for the FCCee luminosity scenarios in
`commondata_projection_FCCee_scenarios/`; systematics are left as a
diagonal-only rescaling of the `commondata_projections_L0/FCCee_240_H_HADR.yaml`
/ `FCCee_365_H_HADR.yaml` covariance, re-diagonalized -- the off-diagonal
correlations are effectively left at their nominal, full-luminosity
strength. The same is true of `commondata_projections_L0/LEP3_240_H_HADR.yaml`,
whose stat errors were rescaled from FCCee_240_H_HADR by the LEP3/FCCee
luminosity ratio at 240 GeV (see the analogous "rescale ... by luminosity
ratio starting from FCCee" pattern in
external_chi2/optimal_observables/interface_oos_lep3.py, applied there to
the 161/240 GeV WW optimal-observable datasets). For these Higgs
hadronic-decay datasets the whole covariance (diagonal AND off-diagonal
terms) is statistics-limited, so every entry should shrink/grow with
luminosity.

Since covmat = sys_add.T @ sys_add for the eigenvector-based decomposition
already stored as the `systematics` block in the nominal L0 files, uniformly
rescaling the full covariance by a factor r is equivalent to rescaling that
whole `systematics` matrix by sqrt(r) -- no need to touch individual entries
or re-diagonalize, unlike `rescale_zdata_systematics.py` (which only
rescales a subset of Z-pole observable columns, since only those are
luminosity limited).

Which files to touch, and the luminosity scaling factor to apply to each,
are set explicitly in TARGET_LUMI_SCALE below -- one entry per target file,
given as a path relative to the repository root, per the table in
commondata_projection_FCCee_scenarios/README.md. This mirrors the
`lumi_scale` argument of rescale_descoped_fccee.py: the systematics factor
applied is sqrt(1 / lumi_scale), the same factor `smefit PROJ` applies to
the statistical errors. Values are always recomputed from the nominal L0
systematics, so the script is safe to run repeatedly.

Usage:
    python scripts/rescale_hadr_systematics.py [--dry-run]
"""

import argparse
import math
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
COMMONDATA_L0 = REPO_ROOT / "commondata_projections_L0"

# Nominal FCCee luminosities, fb^-1 (commondata_projections_L0/FCCee_{240,365}_H_HADR.yaml).
NOMINAL_LUMI_240 = 10800
NOMINAL_LUMI_365 = 3120

NOMINAL_FILES = {
    "240": COMMONDATA_L0 / "FCCee_240_H_HADR.yaml",
    "365": COMMONDATA_L0 / "FCCee_365_H_HADR.yaml",
}

# Target file (relative to REPO_ROOT) -> luminosity scaling factor, i.e.
# target_lumi / nominal_lumi (at the matching energy stage). A factor of 1.0
# means that stage is unaffected (e.g. a top-only luminosity boost).
TARGET_LUMI_SCALE = {
    "commondata_projection_FCCee_scenarios/commondata_projections_fccee_30MW_2IP_36p5lumi/FCCee_240_H_HADR.yaml": 0.365,
    "commondata_projection_FCCee_scenarios/commondata_projections_fccee_30MW_2IP_36p5lumi/FCCee_365_H_HADR.yaml": 0.365,
    "commondata_projection_FCCee_scenarios/commondata_projections_fccee_30MW_4IP_60lumi/FCCee_240_H_HADR.yaml": 0.6,
    "commondata_projection_FCCee_scenarios/commondata_projections_fccee_30MW_4IP_60lumi/FCCee_365_H_HADR.yaml": 0.6,
    "commondata_projection_FCCee_scenarios/commondata_projection_fccee_upscoped_120lumi_global/FCCee_240_H_HADR.yaml": 1.2,
    "commondata_projection_FCCee_scenarios/commondata_projection_fccee_upscoped_120lumi_global/FCCee_365_H_HADR.yaml": 1.2,
    "commondata_projection_FCCee_scenarios/commondata_projection_fccee_upscoped_150lumi_top_only/FCCee_240_H_HADR.yaml": 1.0,
    "commondata_projection_FCCee_scenarios/commondata_projection_fccee_upscoped_150lumi_top_only/FCCee_365_H_HADR.yaml": 1.5,
    # LEP3 (240 GeV): 2304.0 fb^-1 (commondata_projections_L0/LEP3_240_H_HADR.yaml).
    "commondata_projections_L0/LEP3_240_H_HADR.yaml": 2304.0 / NOMINAL_LUMI_240,
}


def _nominal_for(rel_path: str) -> dict:
    """Pick the FCCee_{240,365}_H_HADR nominal matching a target's energy stage."""
    for energy, path in NOMINAL_FILES.items():
        if energy in Path(rel_path).name:
            return yaml.safe_load(path.read_text())
    raise ValueError(f"Could not determine energy stage for {rel_path}")


class _YamlfmtDumper(yaml.Dumper):
    """PyYAML dumper that indents list items to match yamlfmt's style."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow=flow, indentless=False)


def _yaml_dump(data, path: Path) -> None:
    """Write YAML matching yamlfmt's list-indentation style."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            Dumper=_YamlfmtDumper,
            default_flow_style=False,
            sort_keys=False,
            width=10000,
        )


def rescale_file(nominal: dict, path: Path, lumi_scale: float, dry_run: bool) -> None:
    """Rescale one target file's systematics in place."""
    label = path.relative_to(REPO_ROOT)
    factor = math.sqrt(1.0 / lumi_scale)

    if math.isclose(factor, 1.0):
        print(f"  [{label}] lumi_scale=1.0 -- skipping")
        return

    data = yaml.safe_load(path.read_text())
    data["systematics"] = [
        [entry * factor for entry in row] for row in nominal["systematics"]
    ]

    print(f"  [{label}] lumi_scale={lumi_scale}  (systematics factor={factor:.6f})")
    if not dry_run:
        _yaml_dump(data, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the computed scale factors without writing any files",
    )
    args = parser.parse_args()

    for rel_path, lumi_scale in TARGET_LUMI_SCALE.items():
        target = REPO_ROOT / rel_path
        if not target.is_file():
            raise FileNotFoundError(target)
        nominal = _nominal_for(rel_path)
        rescale_file(nominal, target, lumi_scale, args.dry_run)


if __name__ == "__main__":
    main()
