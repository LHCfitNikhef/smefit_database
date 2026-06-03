import yaml
import json
import math

with open("commondata/CMS_DYMee_13TeV.yaml", "r") as file:
    experimental_card = yaml.safe_load(file)

with open("theory/CMS_DYMee_13TeV.json") as file:
    theory_card = json.load(file)

# This is to produce an LO projection for the CMS dataset
# so same lumi and syst unc, but central values (and consequently stat unc) from theory

# update experimental card
experimental_card["data_central"] = theory_card["best_sm"]
experimental_card["statistical_error"] = [
    math.sqrt(x) if x > 1 else 1.0 for x in experimental_card["data_central"]
]

# Save new cards
with open("commondata_projections_L0/CMS_DYMee_13TeV.yaml", "w") as file:
    yaml.dump(experimental_card, file, sort_keys=False)
