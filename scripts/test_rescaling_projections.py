import yaml
import pathlib
import numpy as np

path_to_commondata_scenarios = pathlib.Path("../commondata_projection_FCCee_scenarios")
path_to_baseline = pathlib.Path("../commondata_projections_L0")

expected_scaling_dict = {
    "commondata_projections_fccee_30MW_4IP_60lumi": 0.6,
    "commondata_projections_fccee_30MW_2IP_36p5lumi": 0.365,
    "commondata_projection_fccee_upscoped_150lumi_top_only": 1.5,
    "commondata_projection_fccee_upscoped_120lumi_global": 1.2,
}

# dictionary of flagged projection that do not obey scaling
flagged_projections = {
    "commondata_projections_fccee_30MW_4IP_60lumi": [],
    "commondata_projections_fccee_30MW_2IP_36p5lumi": [],
    "commondata_projection_fccee_upscoped_150lumi_top_only": [],
    "commondata_projection_fccee_upscoped_120lumi_global": [],
}

for scenario, lumi_factor in expected_scaling_dict.items():
    path_to_scenario = path_to_commondata_scenarios / scenario
    for projection in path_to_scenario.iterdir():
        if not projection.name.startswith("FCCee"):
            continue
        else:
            with open(projection, "r") as f:
                data = yaml.safe_load(f)
            stats_scenario = np.array(data["statistical_error"])

            # load full FCC
            with open(path_to_baseline / projection.name, "r") as f:
                data_baseline = yaml.safe_load(f)
            stats_baseline = np.array(data_baseline["statistical_error"])

            # check if the stats obey scaling
            if "top_only" in scenario and "365" not in projection.name:
                if not np.allclose(stats_scenario, stats_baseline, atol=0):
                    flagged_projections[scenario].append(projection.name)
            else:
                if not np.allclose(
                    stats_scenario, stats_baseline / np.sqrt(lumi_factor), rtol=1e-3
                ):
                    flagged_projections[scenario].append(projection.name)

# print summary of flagged projections
for scenario, flagged in flagged_projections.items():
    print(scenario)
    for projection in flagged:
        print(f"  {projection}")
