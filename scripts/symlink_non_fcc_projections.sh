#!/usr/bin/env bash
# Symlink all non-FCC files from commondata_projections_fccee_30MW_4IP_60lumi
# (which serves as the reference for which non-FCC datasets to include) into a
# target directory, using relative paths pointing into commondata_projections_L0.
# Usage: scripts/symlink_non_fcc_projections.sh <target_dir>

SCENARIOS=(
    "commondata_projection_fccee_upscoped_120lumi_global"
    "commondata_projection_fccee_upscoped_150lumi_top_only"
    "commondata_projections_fccee_30MW_2IP_36p5lumi"
    "commondata_projections_fccee_30MW_4IP_60lumi"
)

# Append the target dir (if given and not already in the list above).
if [[ -n "$1" ]]; then
    name="$(basename "$1")"
    if [[ ! " ${SCENARIOS[*]} " == *" $name "* ]]; then
        SCENARIOS+=("$name")
    fi
fi

# symlink all LHC and HL-LHC projections
cd ../commondata_projection_FCCee_scenarios/
for scenario in "${SCENARIOS[@]}"; do
    cd $scenario
    for f in ../../commondata_projections_L0/*.yaml; do

    if [[ "$f" != *ATLAS* && "$f" != *CMS*  && "$f" != *HLLHC* && "$f" != *LEP_* && "$f" != *LEP1*  ]]; then
        continue
    fi
      ln -sf $f .
    done
    cd ../
done

# symlink all non-ttbar FCC data
cd ../commondata_projection_FCCee_scenarios/commondata_projection_fccee_upscoped_150lumi_top_only
for f in ../../commondata_projections_L0/FCCee*.yaml; do
    # if 365 in filename skip
    if [[ "$f" == *365* ]]; then
        continue
    fi
    ln -s $f .
done

