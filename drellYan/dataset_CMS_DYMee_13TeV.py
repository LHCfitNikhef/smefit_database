# Theory card for CMS dataset from arXiv:2103.02708

import yaml
import json
import math 

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


# theory card
theory = {
    'best_sm': [from_events_to_xsec(entry['value']) for entry in mee_background], # cross-check with MATRIX in progress
    'scales': [91.18 for entry in mee_background]
}
 
with open(DATASETNAME + '.json', "w") as outfile:
    outfile.write(json.dumps(theory, indent=2))
    print("Created file: " + DATASETNAME + '.json')
