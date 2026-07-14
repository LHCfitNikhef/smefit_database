# FCCee Projection Scenarios

This directory contains four alternative FCCee luminosity scenarios used for projection studies. Each subdirectory is a complete commondata collection (LHC + FCCee datasets) where the FCCee datasets have been rescaled to a different integrated luminosity relative to the baseline projections in `commondata_projections_L0`.

## Scenario directories

| Directory | Description | Luminosity scaling |
|---|---|---|
| `commondata_projections_fccee_30MW_2IP_36p5lumi` | Descoped: 30 MW RF power, 2 interaction points | × 0.365 (all stages) |
| `commondata_projections_fccee_30MW_4IP_60lumi` | Descoped: 30 MW RF power, 4 interaction points | × 0.6 (all stages) |
| `commondata_projection_fccee_upscoped_120lumi_global` | Upscoped: 20 % more luminosity at all energy stages | × 1.2 (all stages) |
| `commondata_projection_fccee_upscoped_150lumi_top_only` | Upscoped: 50 % more luminosity at the 365 GeV stage only | × 1.5 at 365 GeV, × 1.0 elsewhere |

The luminosity scaling is applied uniformly to statistical errors. Systematic uncertainties and theory covariance matrices are kept consistent with the scenario via the theory JSON keys described below.

Non-FCCee datasets (ATLAS, CMS, HLLHC, LEP) are identical copies of the files in `commondata_projections_L0` and are included so that each subdirectory is a self-contained commondata path for the `smefit` runcard system.

## Theory covariance keys

Theory covariance matrices for FCCee datasets are stored in the theory JSON files under `theory/`. The set of available `theory_cov` keys depends on which dataset is being used.

### Most FCCee datasets

All FCCee datasets except `FCCee_Zdata` carry three theory-uncertainty assumptions that are **luminosity-independent** (the same three keys apply for all four scenarios above):

| Key | Theory uncertainty assumption |
|---|---|
| `theory_cov_aggressive` | Aggressive (smaller) theory uncertainties |
| `theory_cov_current` | Current (central) theory uncertainties |
| `theory_cov_conservative` | Conservative (larger) theory uncertainties |

### FCCee_Zdata (Z-pole EWPOs)

`FCCee_Zdata` (electroweak precision observables at the Z pole, 91 GeV) has **scenario-specific** theory covariance keys because its theory uncertainties depend on both the luminosity scenario and on whether the top quark mass is treated as a free parameter in the fit. These enter through electroweak radiative corrections.

The keys follow the pattern `theory_cov_{type}_{scenario}[_{mt_option}]`:

- **type**: `aggressive`, `current`, or `conservative`
- **scenario**: encodes which luminosity configuration the covariance corresponds to
- **mt_option**: `mt` (top mass floating) or `nomt` (top mass fixed); absent for upscoped scenarios where only the `mt` variant is provided

| Scenario | Commondata source | `mt` key suffix | `nomt` key suffix |
|---|---|---|---|
| Nominal (baseline) | `commondata_projections_L0` | `nominal_mt` | `nominal_nomt` |
| Descoped 36.5 % | `commondata_projections_fccee_30MW_2IP_36p5lumi` | `desc_36p5lumi_mt` | `desc_36p5lumi_nomt` |
| Descoped 60 % | `commondata_projections_fccee_30MW_4IP_60lumi` | `desc_60lumi_mt` | `desc_60lumi_nomt` |
| Upscoped 120 % (global) | `commondata_projection_fccee_upscoped_120lumi_global` | `upscoped_120lumi` | — |
| Upscoped 150 % (top only) | `commondata_projection_fccee_upscoped_150lumi_top_only` | `upscoped_150lumimt` | — |

For example, to use the aggressive theory covariance for the descoped 60 % scenario with the top mass floating, select:

```
theory_cov_aggressive_desc_60lumi_mt
```

### Complete list of valid theory_cov keys for FCCee_Zdata

```
theory_cov_aggressive_nominal_mt
theory_cov_aggressive_nominal_nomt
theory_cov_aggressive_desc_36p5lumi_mt
theory_cov_aggressive_desc_36p5lumi_nomt
theory_cov_aggressive_desc_60lumi_mt
theory_cov_aggressive_desc_60lumi_nomt
theory_cov_aggressive_upscoped_120lumi
theory_cov_aggressive_upscoped_150lumimt

theory_cov_current_nominal_mt
theory_cov_current_nominal_nomt
theory_cov_current_desc_36p5lumi_mt
theory_cov_current_desc_36p5lumi_nomt
theory_cov_current_desc_60lumi_mt
theory_cov_current_desc_60lumi_nomt
theory_cov_current_upscoped_120lumi
theory_cov_current_upscoped_150lumimt

theory_cov_conservative_nominal_mt
theory_cov_conservative_nominal_nomt
theory_cov_conservative_desc_36p5lumi_mt
theory_cov_conservative_desc_36p5lumi_nomt
theory_cov_conservative_desc_60lumi_mt
theory_cov_conservative_desc_60lumi_nomt
theory_cov_conservative_upscoped_120lumi
theory_cov_conservative_upscoped_150lumimt
```
