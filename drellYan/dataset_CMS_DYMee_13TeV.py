# Theory card for CMS dataset from arXiv:2103.02708

import yaml
import json
import math 
import os

# Constants
DATASETNAME = 'CMS_DYMee_13TeV'
LUMINOSITY = 137

# Convert data from events per bin [events/GeV] to differential x-section [fb/GeV]
def from_events_to_xsec(n_events_per_GeV: float) -> float:
    return n_events_per_GeV / LUMINOSITY

# Estimate statistical error
def statistical_error(n_events_per_GeV: float, mee_bin: dict) -> float:
    bin_width = mee_bin['high'] - mee_bin['low']
    
    # if n_events == 0 we estimate error by putting n_events = 1
    n_events = n_events_per_GeV * bin_width if n_events_per_GeV != 0 else 1
    n_events_statistical_error = math.sqrt(n_events)
    
    return n_events_statistical_error / LUMINOSITY / bin_width

# Estimate systematic error
def systematic_error(n_events_per_GeV: float, mee_bin: dict) -> float:
    bin_width = mee_bin['high'] - mee_bin['low']
    selection_efficency = 0.08

    # if n_events == 0 we estimate error by putting n_events = 1
    n_events = n_events_per_GeV * bin_width if n_events_per_GeV != 0 else 1
    n_events_systematic_error = n_events * selection_efficency
    
    return n_events_systematic_error / LUMINOSITY / bin_width

def read_json_result(operators: list) -> list:
    if operators == ["SM"]:
        file_result = "mg5_LO/results_SM.json"
    elif len(operators) == 1:
        file_result = f"mg5_LO/linear/results_{operators[0]}.json"
    elif len(operators) == 2:
        file_result = f"mg5_LO/quadratic/results_{operators[0]}_{operators[1]}.json"

    if not os.path.exists(file_result):
        return operators

    with open(file_result, 'r') as file:
        result = json.load(file)
    
    return result
    

# Reading the dataset downloaded from HEPData 
with open('HEPData-ins1849964-v2-Dielectron_mass_distribution.yaml', 'r') as file:
    dataset_CMS = yaml.safe_load(file)

# Extracting data, keeping only the ones for m_ee > 500 GeV
mee_bins = dataset_CMS['independent_variables'][0]['values'][38:]           
mee_values = dataset_CMS['dependent_variables'][0]['values'][38:]
mee_background = dataset_CMS['dependent_variables'][1]['values'][38:]

data_central = [from_events_to_xsec(entry['value']) for entry in mee_values]
statistical_error = [statistical_error(entry['value'], bin_mee) for entry, bin_mee in zip(mee_values,mee_bins)]
systematic_error = [[systematic_error(entry['value'], bin_mee) for entry, bin_mee in zip(mee_values,mee_bins)]]

# commondata file
commondata ={
    'dataset_name': DATASETNAME,
    'doi': 'https://doi.org/10.17182/hepdata.101186.v2',
    'location': 'Data from Figure 2 left, bins for m_ee > 200 GeV',
    'arxiv': 2103.02708,
    'hepdata': 'https://www.hepdata.net/record/ins1849964?version=2&table=Dielectron%20mass%20distribution',
    'units': 'fb/GeV',
    'description': 'The invariant mass distribution for electron pair production in the range 500-6070 GeV.',
    'luminosity': LUMINOSITY, # units: fb^-1
    'num_data': len(mee_bins),
    'num_sys': 1,
    'data_central': data_central,
    'statistical_error': statistical_error,
    'systematics': systematic_error,
    'sys_names': 'UNCORR',
    'sys_type': 'ADD'
}

with open(DATASETNAME + '.yaml', 'w') as file:
    yaml.dump(commondata, file, sort_keys=False)
    print("Created file: " + DATASETNAME + '.yaml')

operators_map = {
    'OpWB': 'OpWB',
    'OpD': 'OpD',
    'Ope': 'Ope',
    'Opmu': 'Opmu',
    'Opl1': 'Opl1',
    'Opl2': 'Opl2',
    'O3pl1': 'O3pl1',
    'O3pl2': 'O3pl2',
    'Opui': 'Opui',
    'Opdi': 'Opdi',
    'OpqMi_small': 'OpqMi',
    'O3pq': 'O3pq',
    'OpQMi_big': 'OpQMi',
    'OpQ3': 'OpQ3',
    'Oeu': 'Oeu',
    'Oed': 'Oed',
    'Olq1': 'Olq1',
    'Olq3': 'Olq3',
    'OQl1': 'OQl1',
    'OQl3': 'OQl3',
    'Olu': 'Olu',
    'Old': 'Old',
    'Oqe': 'Oqe',
    'Oll': 'Oll'
}
operators = list(operators_map.keys())

# theory card
theory = {
    'best_sm': [from_events_to_xsec(entry['value']) for entry in mee_background], 
    'scales': [91.18 for entry in mee_background],
    'LO': {
        'SM': read_json_result(['SM']),
        **{operators_map[op]: read_json_result([op]) for op in operators},
        **{f"{operators_map[operators[i]]}*{operators_map[operators[j]]}": read_json_result([operators[i], operators[j]]) for i in range(len(operators)) for j in range(i, len(operators))}
    }
}
 
with open(DATASETNAME + '.json', "w") as outfile:
    outfile.write(json.dumps(theory, indent=2))
    print("Created file: " + DATASETNAME + '.json')


