import numpy as np


class OptimalWW:

    def __init__(self, coefficients):

        oo_wc_basis = ['OpD', 'OpWB', 'OWWW', 'Opl1', 'Ope', 'O3pl1']

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            'FCC_ee_ww_lepto_161': '/data/theorie/jthoeve/smefit_database/external_chi2/invcov_FCCee_ww_leptonic_161.dat',
            'FCC_ee_ww_lepto_240': '/data/theorie/jthoeve/smefit_database/external_chi2/invcov_FCCee_ww_leptonic_240.dat',
            'FCC_ee_ww_lepto_365': '/data/theorie/jthoeve/smefit_database/external_chi2/invcov_FCCee_ww_leptonic_365.dat',
            'FCC_ee_ww_semilep_161': '/data/theorie/jthoeve/smefit_database/external_chi2/invcov_FCCee_ww_semilep_161.dat',
            'FCC_ee_ww_semilep_240': '/data/theorie/jthoeve/smefit_database/external_chi2/invcov_FCCee_ww_semilep_240.dat',
            'FCC_ee_ww_semilep_365': '/data/theorie/jthoeve/smefit_database/external_chi2/invcov_FCCee_ww_semilep_365.dat'
        }

        self.n_dat = len(oo_wc_basis)

    def compute_chi2(self, coefficient_values):

        chi2_value = 0
        for dataset, invcov_path in self.datasets.items():
            invcov = np.loadtxt(invcov_path)
            chi2_value += np.linalg.multi_dot([coefficient_values, self.project.T, invcov, self.project, coefficient_values])

        return chi2_value


class Optimaltt:

    def __init__(self, coefficients):

        oo_tt_wc_basis = ['OpQM', 'Opt', 'OtW', 'OtZ']

        self.project = np.zeros((len(oo_tt_wc_basis), coefficients.size))
        for i, op in enumerate(oo_tt_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            'FCC_ee_tt_365': '/data/theorie/jthoeve/smefit_database/external_chi2/invcov_FCC_ee_tt_365GeV.dat'
        }
        self.n_dat = len(oo_tt_wc_basis)

    def compute_chi2(self, coefficient_values):

        chi2_value = 0
        for dataset, invcov_path in self.datasets.items():
            invcov = np.loadtxt(invcov_path)
            chi2_value += np.linalg.multi_dot([coefficient_values, self.project.T, invcov, self.project, coefficient_values])
        return chi2_value






