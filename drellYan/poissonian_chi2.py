import sys
import numpy as np
from smefit import loader

class ExternalChi2:
    def __init__(self, coefficients):
        """
        Constructor that allows one to set attributes that can be called in the compute_chi2 method
        Parameters
        ----------
        coefficients:  smefit.coefficients.CoefficientManager
            attributes: name, value
        """

        # SMEFT operators that appear in LHC + DY fit
        operators={'OpQM': {'min': -5, 'max': 5}, 'O3pQ3': {'min': -2, 'max': 2}, 'Opt': {'min': -10.0, 'max': 10.0}, 'OtW': {'min': -1.0, 'max': 1.0}, 'OtG': {'min': -1.0, 'max': 1.0}, 'Otp': {'min': -10.0, 'max': 10.0}, 'OtZ': {'min': -15, 'max': 15}, 'OQQ1': {'min': -10, 'max': 10.0}, 'OQQ8': {'min': -20.0, 'max': 20.0}, 'OQt1': {'min': -10.0, 'max': 10.0}, 'OQt8': {'min': -20.0, 'max': 20.0}, 'Ott1': {'min': -10.0, 'max': 10.0}, 'O81qq': {'min': -3, 'max': 3}, 'O11qq': {'min': -12, 'max': 12}, 'O83qq': {'min': -10, 'max': 10}, 'O13qq': {'min': -1, 'max': 1}, 'O8qt': {'min': -7, 'max': 7}, 'O1qt': {'min': -11, 'max': 11}, 'O8ut': {'min': -12, 'max': 12}, 'O1ut': {'min': -30, 'max': 30}, 'O8qu': {'min': -15, 'max': 15}, 'O1qu': {'min': -15, 'max': 15}, 'O8dt': {'min': -30, 'max': 30}, 'O1dt': {'min': -50, 'max': 50}, 'O8qd': {'min': -20, 'max': 20}, 'O1qd': {'min': -30, 'max': 30}, 'OpG': {'min': -0.1, 'max': 0.1}, 'OpB': {'min': -1, 'max': 1}, 'OpW': {'min': -1, 'max': 1}, 'OpWB': {'min': -2, 'max': 2}, 'Opd': {'min': -5.0, 'max': 5.0}, 'OpD': {'min': -2, 'max': 2.0}, 'OpqMi': {'min': -1, 'max': 1}, 'O3pq': {'min': -1, 'max': 1.0}, 'Opui': {'min': -1, 'max': 1.0}, 'Opdi': {'min': -1, 'max': 1.0}, 'Ocp': {'min': -1, 'max': 1}, 'Obp': {'min': -0.2, 'max': 0.2}, 'Opl1': {'min': -1, 'max': 1.0}, 'Opl2': {'min': -0.5, 'max': 0.5}, 'Opl3': {'min': -0.5, 'max': 0.5}, 'O3pl1': {'min': -0.5, 'max': 0.5}, 'O3pl2': {'min': -0.5, 'max': 0.5}, 'O3pl3': {'min': -0.5, 'max': 0.5}, 'Ope': {'min': -1, 'max': 1.0}, 'Opmu': {'min': -1, 'max': 1.0}, 'Opta': {'min': -1, 'max': 1.0}, 'Otap': {'min': -0.2, 'max': 0.2}, 'Oll': {'min': -0.5, 'max': 0.5}, 'Oll1111': {'constrain': [{'Oll': 1}], 'min': -1.5, 'max': 1.0}, 'OWWW': {'min': -1, 'max': 1}, 'Oeu': {'min': -20, 'max': 20}, 'Oed': {'min': -20, 'max': 20}, 'Oeb': {'min': -20, 'max': 20}, 'Olq1': {'min': -20, 'max': 20}, 'Olq3': {'min': -20, 'max': 20}, 'OQl1': {'min': -20, 'max': 20}, 'OQl3': {'min': -20, 'max': 20}, 'Olu': {'min': -20, 'max': 20}, 'Old': {'min': -20, 'max': 20}, 'Olb': {'min': -20, 'max': 20}, 'Oqe': {'min': -20, 'max': 20}, 'OQe': {'min': -20, 'max': 20}}
    
        # Importing drell_yan dataset using smefit built-in function
        self.drell_yan_dataset = loader.load_datasets(
            commondata_path='./commondata',
            datasets=['CMS_DYMee_13TeV'],
            operators_to_keep=operators,
            order='NLO',
            use_quad=True,
            use_theory_covmat=True,
            use_t0=False,
            use_multiplicative_prescription=False,
            theory_path='./theory',
            rot_to_fit_basis=None,
            has_uv_couplings=False,
            has_external_chi2={'ExternalChi2': '/Users/armadillo/Desktop/PhD/smefit/smefit_database/runcards/poissonian_chi2.py'}
        )
        
    def compute_theory_predictions(self, coefficient_values):
        sm = self.drell_yan_dataset.SMTheory
        
        # linear contributions
        linear = np.matmul(self.drell_yan_dataset.LinearCorrections, coefficient_values)

        # quadratic contributions
        wc_coeff_sq = np.array( [coefficient_values[i] * coefficient_values[j] for i in range(len(coefficient_values)) for j in range(i, len(coefficient_values))] )
        quadratic = np.matmul(self.drell_yan_dataset.QuadraticCorrections, wc_coeff_sq)

        return sm + linear + quadratic

    def compute_chi2(self, coefficient_values):
        """
        Parameters
        ----------
         coefficients_values : numpy.ndarray
            |EFT| coefficients values

        """
        theory_predictions = self.compute_theory_predictions(coefficient_values)

        # Standard gaussian chi2, used only for debug
        # diff = self.drell_yan_dataset.Commondata - theory_predictions
        # chi2 = diff.T @ self.drell_yan_dataset.InvCovMat @ diff
    
        # Log-likelihood for a poissonian distribution, is the log of the likelihood given in Eq. 4.10 in 2104.02723
        # The overall minus sign is because smefit tries to minimize this quantity
        chi2 = - np.sum( self.drell_yan_dataset.Commondata * np.log(theory_predictions) - theory_predictions )

        return chi2
