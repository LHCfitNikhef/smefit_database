# Theory card for CMS dataset from arXiv:2103.02708

import yaml
import json
import math 
import os
import numpy as np

# Constants
DATASETNAME = 'CMS_DYMee_13TeV'
LUMINOSITY = 137

# Convert data from events per bin [events/GeV] to differential x-section [fb/GeV]
def from_events_per_GeV_to_events(n_events_per_GeV: float, mee_bin: dict) -> float:
    bin_width = mee_bin['high'] - mee_bin['low']
    return n_events_per_GeV * bin_width

# Estimate statistical error
def statistical_error(n_events: float) -> float:
    # if n_events == 0 we estimate error by putting n_events = 1
    n_events_corr = n_events if n_events > 1 else 1.
    n_events_statistical_error = math.sqrt(n_events_corr)

    return n_events_statistical_error

# Estimate systematic error
def systematic_error(n_events: float) -> float:
    selection_efficency = 0.08

    # if n_events == 0 we estimate error by putting n_events = 1
    n_events_corr = n_events if n_events > 1 else 1.
    n_events_systematic_error = n_events_corr * selection_efficency
    
    return n_events_systematic_error 

def read_json_result(operators: list, mee_width: list, order: str, acceptance_lst: list) -> list:
    if operators == ["SM"]:
        file_result = f"mg5_{order}/results_SM_{order}.json"
    elif len(operators) == 1:
        file_result = f"mg5_{order}/linear/results_{operators[0]}.json"
    elif len(operators) == 2:
        file_result = f"mg5_{order}/quadratic/results_{operators[0]}_{operators[1]}.json"

    if not os.path.exists(file_result):
        return operators

    with open(file_result, 'r') as file:
        result = json.load(file)
    
    nevents = [dsigmadmee * width * LUMINOSITY * acc for dsigmadmee, width, acc in zip(result,mee_width,acceptance_lst)]
    return nevents
    

# Reading the dataset downloaded from HEPData 
with open('HEPData-ins1849964-v2-Dielectron_mass_distribution.yaml', 'r') as file:
    dataset_CMS = yaml.safe_load(file)

with open('HEPData-ins1849964-v2-Acceptance_x_Efficiency_as_a_function_of_dielectron_mass.yaml') as file:
    acceptance_CMS = yaml.safe_load(file)

# Processing acceptance
energy_tmp = acceptance_CMS['independent_variables'][0]['values']
acceptance_tmp = acceptance_CMS['dependent_variables'][0]['values']
mee_acceptance = [(en['value'], acc['value']) for en, acc in zip(energy_tmp, acceptance_tmp)]

# Extracting data, keeping only the ones for m_ee > 500 GeV
mee_bins = dataset_CMS['independent_variables'][0]['values'][38:] 
mee_width = [mee_bin['high'] - mee_bin['low'] for mee_bin in mee_bins]          
mee_values = dataset_CMS['dependent_variables'][0]['values'][38:]
mee_background = dataset_CMS['dependent_variables'][1]['values'][38:]

data_central = [from_events_per_GeV_to_events(entry['value'],bin_mee) for entry, bin_mee in zip(mee_values,mee_bins)]

# Estimate background which is not DY

# My best prediction is SM NLO QCD * acceptance
acceptance_lst = []
for mee_bin in mee_bins:
    acceptance_tmp = [] 
    for acceptance in mee_acceptance:
        if acceptance[0] < mee_bin['low']:
            continue
        elif acceptance[0] > mee_bin['high']:
            break
        acceptance_tmp.append(acceptance[1])

    # After 5500 GeV remains constant
    if acceptance_tmp == []:
        acceptance_tmp.append(0.60596)

    acceptance_lst.append(float(np.mean(acceptance_tmp)))

my_best_DY = read_json_result(['SM'], mee_width, 'NLO', acceptance_lst)

# The best simulation (DY+top+diboson) is provided by CMS
best_SM_by_CMS = [from_events_per_GeV_to_events(entry['value'], mee_bin) for entry, mee_bin in zip(mee_background, mee_bins)]

# Estimate the background: best_CMS_simulation - my_best_DY_prediction
background_estimate = [sm - dy if sm>dy else 0 for sm, dy in zip(best_SM_by_CMS, my_best_DY)]

# Only DY data = total data - background estimate
data_only_DY = [data - background if data>background else 0 for data, background in zip(data_central, background_estimate)]

# Estimate statistical and systematic error
statistical_error = [statistical_error(entry) for entry in data_only_DY]
systematic_error = [[systematic_error(entry) for entry in data_only_DY]]

# commondata file
commondata ={
    'dataset_name': DATASETNAME,
    'doi': 'https://doi.org/10.17182/hepdata.101186.v2',
    'location': 'Data from Figure 2 left, bins for m_ee > 500 GeV',
    'arxiv': 2103.02708,
    'hepdata': 'https://www.hepdata.net/record/ins1849964?version=2&table=Dielectron%20mass%20distribution',
    'units': '# events',
    'description': 'The number of events observed for electron pair production in the range 500-6070 GeV.',
    'luminosity': LUMINOSITY, # units: fb^-1
    'num_data': len(mee_bins),
    'num_sys': 1,
    'data_central': data_only_DY,
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
    'Opl1': 'Opl1',
    'O3pl1': 'O3pl1',
    'Opui': 'Opui',
    'Opdi': 'Opdi',
    'Opb': 'Obp',
    'OpqMi_small': 'OpqMi',
    'O3pq': 'O3pq',
    'OpQMi_big': 'OpQM',
    'OpQ3': 'O3pQ3',
    'Oeu': 'Oeu',
    'Oed': 'Oed',
    'Oeb': 'Oeb',
    'Olq1': 'Olq1',
    'Olq3': 'Olq3',
    'OQl1': 'OQl1',
    'OQl3': 'OQl3',
    'Olu': 'Olu',
    'Old': 'Old',
    'Olb': 'Olb',
    'Oqe_small': 'Oqe',
    'OQe_big': 'OQe',
    'Oll': 'Oll'
}
operators = list(operators_map.keys())

# theory card
theory = {
    'best_sm': my_best_DY, 
    'scales': [(tmp_bin['high'] + tmp_bin['low'])/2. for tmp_bin in mee_bins],
    "theory_cov": [ [0 for j in mee_background] for i in mee_background],
    'LO': {
        'SM': read_json_result(['SM'], mee_width, 'LO', acceptance_lst),
        **{operators_map[op]: read_json_result([op], mee_width, 'LO', acceptance_lst) for op in operators},
        **{f"{operators_map[operators[i]]}*{operators_map[operators[j]]}": read_json_result([operators[i], operators[j]], mee_width, 'LO', acceptance_lst) for i in range(len(operators)) for j in range(i, len(operators))}
    },
    'NLO': {
        'SM': read_json_result(['SM'], mee_width, 'NLO', acceptance_lst),
        **{operators_map[op]: read_json_result([op], mee_width, 'LO', acceptance_lst) for op in operators},
        **{f"{operators_map[operators[i]]}*{operators_map[operators[j]]}": read_json_result([operators[i], operators[j]], mee_width, 'LO', acceptance_lst) for i in range(len(operators)) for j in range(i, len(operators))}
    }
}
 
with open(DATASETNAME + '.json', "w") as outfile:
    outfile.write(json.dumps(theory, indent=2))
    print("Created file: " + DATASETNAME + '.json')
