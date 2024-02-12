import pathlib
import yaml

database_path = pathlib.Path('./commondata/')
lumi_threshold = 50
datasets_proj = {240: [], 365: []}

for dataset in database_path.iterdir():

    # skip FCC data
    skip_dataset = not dataset.stem.startswith('FCC')
    if skip_dataset:
        continue
    else:
        for com in datasets_proj.keys():
            if str(com) in dataset.stem:
                datasets_proj[com].append(dataset.stem)




with open('datasets_fcc.dat', 'w') as f:

    f.writelines('\n'.join(datasets_proj[240]) + '\n')
    f.writelines('\n'.join(datasets_proj[365]) + '\n')

# check whether breakdown is avaible. 