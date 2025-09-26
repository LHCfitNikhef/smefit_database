import yaml
import json
import math

with open("commondata/CMS_DYMee_13TeV.yaml", "r") as file:
    experimental_card = yaml.safe_load(file)

with open("theory/CMS_DYMee_13TeV.json") as file:
    theory_card = json.load(file)

# rescale theory files
theory_card["best_sm"] = [(3000.0 / 137.0) * x for x in theory_card["best_sm"]]

lo_keys = theory_card["LO"].keys()
for key in lo_keys:
    theory_card["LO"][key] = [(3000.0 / 137.0) * x for x in theory_card["LO"][key]]

nlo_keys = theory_card["NLO_QCD"].keys()
for key in nlo_keys:
    theory_card["NLO_QCD"][key] = [
        (3000.0 / 137.0) * x for x in theory_card["NLO_QCD"][key]
    ]

# update experimental card
experimental_card["dataset_name"] = "HLLHC_DYMee_13TeV"
experimental_card["luminosity"] = 3000
experimental_card["data_central"] = theory_card["best_sm"]
experimental_card["statistical_error"] = [
    math.sqrt(x) if x > 1 else 1.0 for x in experimental_card["data_central"]
]
experimental_card["systematics"] = [
    [x / 2.0 for x in experimental_card["systematics"][0]]
]

# Save new cards
with open("commondata_projections_L0/HLLHC_DYMee_13TeV.yaml", "w") as file:
    yaml.dump(experimental_card, file, sort_keys=False)

with open("theory/HLLHC_DYMee_13TeV.json", "w") as file:
    file.write(json.dumps(theory_card, indent=2))
