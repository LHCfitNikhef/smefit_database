import json
import yaml
import numpy as np
import pathlib


# Some numerical global constants
mt = 172.0
mh = 125.0
mz = 91.1876
mw = 80.387


def scale_ttZ(bin):
    return np.sqrt((2 * mt + mz) ** 2 + bin ** 2) # bin = pT_V

def scale_ggF(bin):
    return np.sqrt(mh ** 2 + bin ** 2) # bin = pT_H

# collect all scale choices in dictionary with process name as key
scale_funct_dict = {'ttZ': scale_ttZ, 'ggF': scale_ggF}

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
    process = dataset_name.split('_')[1]

    if 'bins' in commondata:
        kin = commondata["kinematic"]
        scales = []
        for bin in commondata['bins']:
            min = bin[kin]['min']
            max = bin[kin]['max']
            bin_average = (min + max) / 2
            scale = scale_funct_dict[process](bin_average)
            scales.append(scale)

        return scales


scales_dict = {}
path_to_commondata = pathlib.Path('../commondata')
for commondata_file in path_to_commondata.iterdir():
    with open(commondata_file, encoding="utf-8") as f:
        commondata = yaml.safe_load(f)

        scales = compute_scale(commondata)
        scales_dict[commondata['dataset_name']] = scales



