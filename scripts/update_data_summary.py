import pathlib
import json
import re


from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq, CommentedMap

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.default_flow_style = None


commondata_path = pathlib.Path("../commondata_projections_L0")
theory_path = pathlib.Path("../theory")

experiments = [
    "LEP",
    "ATLAS_CMS",
    "ATLAS",
    "ATLAS_uncor",
    "CMS",
    "CMS_uncor",
    "HLLHC_proj",
    "FCCee",
    "CEPC",
]

# Root mapping
data_summary = {
    "Info": "Collection of all data implemented in SMEFiT, including the available theory options."
}

for commondata_file in sorted(commondata_path.iterdir()):
    if commondata_file.is_file() and commondata_file.suffix == ".yaml":
        dataset_name = commondata_file.stem

        try:
            with open(
                theory_path / f"{dataset_name}.json", "r", encoding="utf-8"
            ) as file:
                theory_content = json.load(file)
        except FileNotFoundError:
            print(f"Warning: Theory file for {dataset_name} not found.")
            continue

        # Decide order
        available_orders = [k for k in theory_content.keys() if "LO" in k]
        if "NLO_EW_only_for_ZH" in theory_content:
            order = "NLO_EW_only_for_ZH"
        elif "NLO_EW" in theory_content:
            order = "NLO_EW"
        elif "NLO_QCD" in theory_content:
            order = "NLO_QCD"
        else:
            order = "LO"

        # Match dataset name to experiments
        for exp in experiments:
            if exp in dataset_name:
                if exp == "FCCee" or exp == "CEPC":
                    try:
                        sqrts = re.findall(r"\d{2,}", dataset_name)[0]
                    except IndexError:
                        sqrts = "91"
                    exp_name = f"{exp}_{sqrts}"
                else:
                    exp_name = exp

                if exp_name not in data_summary:
                    data_summary[exp_name] = CommentedSeq()

                seq = data_summary[exp_name]

                # Create item and append
                item = CommentedMap(name=dataset_name, order=order)
                seq.append(item)
                seq.yaml_add_eol_comment(
                    "order: " + ", ".join(f"{order}" for order in available_orders),
                    len(seq) - 1,
                )


with open("../data_summary.yaml", "w") as f:
    yaml.dump(data_summary, f)
