import pathlib
import yaml

database_path = pathlib.Path('./commondata/')
lumi_threshold = 50
datasets_proj = {}
datasets_no_breakdown = []

for dataset in database_path.iterdir():

    # skip FCC data
    skip_dataset = dataset.stem.startswith('FCC') or dataset.stem.startswith('LEP') or '13' not in dataset.stem
    if skip_dataset:
        continue
    else:
        with open(dataset) as f:
            dataset_loaded = yaml.safe_load(f)
        if "luminosity" in dataset_loaded:
            lumi = dataset_loaded["luminosity"]
        else:
            print("No luminosity specified, skipping dataset {}".format(dataset.stem))

        if "statistical_error" in dataset_loaded:
            stat_error = dataset_loaded["statistical_error"]
            if isinstance(stat_error, list):
                if stat_error[0] == 0:
                    datasets_no_breakdown.append(dataset.stem)
                    continue
            else:
                if stat_error == 0:
                    datasets_no_breakdown.append(dataset.stem)
                    continue

        if lumi > lumi_threshold:
            datasets_proj[dataset.stem] = lumi


# with open('datasets_proj.yaml', 'w') as f:
#     for dataset, lumi in datasets_proj.items():
#         dataset_escp = dataset.replace('_', '\_')
#         f.write(fr'{{\tt {dataset_escp}}}&${lumi}$&&&&\\' + '\n')

with open('hllhc_datasets.dat', 'w') as f:
    f.writelines('\n'.join(list(datasets_proj.keys())) + '\n')