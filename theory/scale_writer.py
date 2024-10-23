import json
import yaml
import numpy as np
import pathlib


# Some numerical global constants
mt = 172.0
mh = 125.0
mz = 91.1876
mw = 80.387


def scale_ttZ(bin, kinematic=None):
    return np.sqrt((2 * mt + mz) ** 2 + bin ** 2) # bin = pT_V

def scale_ggF(bin, kinematic=None):
    return np.sqrt(mh ** 2 + bin ** 2) # bin = pT_H

def scale_H_VBF(bin, kinematic=None):
    return scale_ggF(bin)

def scale_tt(bin, kinematic=None):
    return bin

def scale_wz(bin, kinematic=None):
    return np.sqrt((mz + mw) ** 2 + bin ** 2)

def scale_tz(bin, kinematic=None):
    if kinematic == 'pTt':
        return np.sqrt((mt + mz) ** 2 + bin ** 2)
    return mt + mz

def scale_tw(bin, kinematic=None):
    return mt + mw

# collect all scale choices in dictionary with process name as key
scale_funct_dict = {'ttZ': scale_ttZ,
                    'ggF': scale_ggF,
                    'H': scale_ggF,
                    'tt2D': scale_tt,
                    'tt': scale_tt,
                    'WZ': scale_wz,
                    'tZ': scale_tz}

def compute_scale(commondata):
    """
    Computes the renormalisation scale of the process in commondata
    Parameters
    ----------
    commondata: dict
        The commondata

    Returns
    -------
    scales: list
        List of the renormalisation scales
    """
    dataset_name = commondata['dataset_name']
    dataset_parts = dataset_name.split('_')
    process = dataset_parts[1]

    #extract kinematic
    kinematic = dataset_parts[-1]
    if kinematic == 'proj':
        kinematic = dataset_parts[-2]

    if 'bins' in commondata:
        kin = commondata["kinematic"]
        scales = []
        for bin in commondata['bins']:
            min = bin[kin]['min']
            max = bin[kin]['max']
            bin_average = (min + max) / 2
            scale = scale_funct_dict[process](bin_average, kinematic)
            scales.append(scale)

        return scales


scales_dict = {}
path_to_commondata = pathlib.Path('../commondata')
theory_base_path = pathlib.Path('../theory')
for commondata_file in path_to_commondata.iterdir():
    with open(commondata_file, encoding="utf-8") as f:
        commondata = yaml.safe_load(f)

        if "bins" in commondata:
            scales = compute_scale(commondata)

            path_to_theory = theory_base_path / f"{commondata_file.stem}.json"

            # add scales to theory file
            with open(path_to_theory) as f_theory:
                theory_config = json.load(f_theory)
                theory_config['scales'] = scales

            # dump theory file
            with open(path_to_theory, "w") as f:
                json.dump(theory_config, f, indent=1)

            scales_dict[commondata['dataset_name']] = scales




