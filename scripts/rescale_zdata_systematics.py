"""Rescale selected Z-pole EWPO systematic uncertainties in FCCee/LEP3 files.

`rescale_descoped_fccee.py` (and the `smefit PROJ` machinery it wraps) only
rescales statistical uncertainties for the FCCee luminosity scenarios in
`commondata_projection_FCCee_scenarios/`; systematics are left as plain
copies of the `commondata_projections_L0/FCCee_Zdata.yaml` values. The same
is true of `commondata_projections_L0/LEP3_Zdata.yaml`, whose stat errors
were rescaled from FCCee_Zdata by the LEP3/FCCee luminosity ratio at 91 GeV
(see the analogous "rescale ... by luminosity ratio starting from FCCee"
pattern in external_chi2/optimal_observables/interface_oos_lep3.py, applied
there to the 161/240 GeV WW optimal-observable datasets). For a subset of
Z-pole observables, the dominant "systematic" is really a statistics-limited
calibration and should shrink/grow with luminosity the same way the
statistical error does. This script applies that rescaling to those
observables only, in each target file's FCCee_Zdata-format commondata.

Rescaled observables: Re, Rmu, Rtau, Ae, Amu, Atau, Rb, Rc, Ab, Ac.
Left untouched: GammaZ, SigmaHad, alphaEW(mZ).

Which files to touch, and the FCCee_91 (Z-pole) luminosity scaling factor
to apply to each, are set explicitly in TARGET_LUMI_SCALE below -- one entry
per target file, given as a path relative to the repository root. This
mirrors the `lumi_scale` argument of rescale_descoped_fccee.py: the
systematics factor applied is sqrt(1 / lumi_scale), the same factor
`smefit PROJ` applies to the statistical errors. Values are always
recomputed from the nominal L0 FCCee_Zdata systematics, so the script is
safe to run repeatedly.

Usage:
    python scripts/rescale_zdata_systematics.py [--dry-run]
"""

import argparse
import math
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
NOMINAL_FILE = REPO_ROOT / "commondata_projections_L0" / "FCCee_Zdata.yaml"

# Nominal FCCee_91 (Z-pole) luminosity, fb^-1 (commondata_projections_L0/FCCee_Zdata.yaml).
NOMINAL_LUMI_91 = 205000

# Target file (relative to REPO_ROOT) -> FCCee_91 luminosity scaling factor,
# i.e. target_lumi / nominal_lumi. A factor of 1.0 means the Z-pole stage is
# unaffected (e.g. a top-only luminosity boost).
TARGET_LUMI_SCALE = {
    "commondata_projection_FCCee_scenarios/commondata_projections_fccee_30MW_2IP_36p5lumi/FCCee_Zdata.yaml": 0.365,
    "commondata_projection_FCCee_scenarios/commondata_projections_fccee_30MW_4IP_60lumi/FCCee_Zdata.yaml": 0.6,
    "commondata_projection_FCCee_scenarios/commondata_projection_fccee_upscoped_120lumi_global/FCCee_Zdata.yaml": 1.2,
    "commondata_projection_FCCee_scenarios/commondata_projection_fccee_upscoped_150lumi_top_only/FCCee_Zdata.yaml": 1.0,
    # LEP3 (91 GeV): 48000 fb^-1 (commondata_projections_L0/LEP3_Zdata.yaml).
    "commondata_projections_L0/LEP3_Zdata.yaml": 48000 / NOMINAL_LUMI_91,
}

# Column ordering in FCCee_Zdata.yaml, per its `description` field:
# GammaZ, SigmaHad, Re, Rmu, Rtau, Ae, Amu, Atau, Rb, Rc, Ab, Ac, alphaEW(mZ)
OBSERVABLE_INDICES = {
    "GammaZ": 0,
    "SigmaHad": 1,
    "Re": 2,
    "Rmu": 3,
    "Rtau": 4,
    "Ae": 5,
    "Amu": 6,
    "Atau": 7,
    "Rb": 8,
    "Rc": 9,
    "Ab": 10,
    "Ac": 11,
    "alphaEW": 12,
}
RESCALED_OBSERVABLES = [
    "Re",
    "Rmu",
    "Rtau",
    "Ae",
    "Amu",
    "Atau",
    "Rb",
    "Rc",
    "Ab",
    "Ac",
]
TARGET_INDICES = [OBSERVABLE_INDICES[name] for name in RESCALED_OBSERVABLES]


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
    for row, nominal_row in zip(data["systematics"], nominal["systematics"]):
        for idx in TARGET_INDICES:
            row[idx] = nominal_row[idx] * factor

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
    parser.add_argument(
        "--target-dir",
        type=Path,
        help="Additional scenario directory whose FCCee_Zdata.yaml should be "
        "rescaled, given together with --lumi-scale. Processed on top of the "
        "entries already listed in TARGET_LUMI_SCALE above.",
    )
    parser.add_argument(
        "--lumi-scale",
        type=float,
        help="Luminosity scale factor to apply to --target-dir's file",
    )
    args = parser.parse_args()

    if (args.target_dir is None) != (args.lumi_scale is None):
        parser.error("--target-dir and --lumi-scale must be given together")

    targets = dict(TARGET_LUMI_SCALE)
    if args.target_dir is not None:
        path = args.target_dir / "FCCee_Zdata.yaml"
        rel = path.resolve().relative_to(REPO_ROOT.resolve())
        targets[str(rel)] = args.lumi_scale

    nominal = yaml.safe_load(NOMINAL_FILE.read_text())

    for rel_path, lumi_scale in targets.items():
        target = REPO_ROOT / rel_path
        if not target.is_file():
            raise FileNotFoundError(target)
        rescale_file(nominal, target, lumi_scale, args.dry_run)


if __name__ == "__main__":
    main()
