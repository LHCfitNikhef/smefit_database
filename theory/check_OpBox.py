import os
import json
import numpy as np

# import every json file in the current directory
for filename in os.listdir("."):
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)

        xs = data["LO"]
        sm = np.array(xs["SM"])
        if "OpBox" in xs:
            op = np.array(xs["OpBox"])

            ratio = op / sm
            # check if ratio is close to 0.12
            if not np.allclose(ratio, 0.12, atol=1e-2):
                print(f"{filename}, order LO: {op / sm}")
        else:
            continue

        if "NLO_QCD" in data:
            xs = data["NLO_QCD"]
            sm = np.array(xs["SM"])
            if "OpBox" in xs:
                op = np.array(xs["OpBox"])

                ratio = op / sm
                # check if ratio is close to 0.12
                if not np.allclose(ratio, 0.12, atol=1e-2):
                    print(f"{filename}, order NLO_QCD: {op / sm}")
            else:
                continue
        if "NLO_EW" in data:
            xs = data["NLO_EW"]
            sm = np.array(xs["SM"])
            if "OpBox" in xs:
                op = np.array(xs["OpBox"])

                ratio = op / sm
                # check if ratio is close to 0.12
                if not np.allclose(ratio, 0.12, atol=1e-2):
                    print(f"{filename}, order NLO_EW: {op / sm}")
            else:
                continue
