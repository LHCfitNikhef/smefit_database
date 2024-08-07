import numpy as np
import pathlib

current_file_path = pathlib.Path(__file__).resolve().parent

# future colliders to include
collider = 'CEPC'

class OptimalWW:

    def __init__(self, coefficients):

        oo_wc_basis = ['OpD', 'OpWB', 'OWWW', 'Opl1', 'Ope', 'O3pl1']

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            '{collider}_ww_lepto_161': 'invcov_{collider}_ww_leptonic_161.dat',
            '{collider}_ww_lepto_240': 'invcov_{collider}_ww_leptonic_240.dat',
            '{collider}_ww_lepto_365':  'invcov_{collider}_ww_leptonic_365.dat',
            '{collider}_ww_semilep_161': 'invcov_{collider}_ww_semilep_161.dat',
            '{collider}_ww_semilep_240':  'invcov_{collider}_ww_semilep_240.dat',
            '{collider}_ww_semilep_365': 'invcov_{collider}_ww_semilep_365.dat'
        }
        
        self.n_dat = len(oo_wc_basis)

    def compute_chi2(self, coefficient_values):

        chi2_value = 0
        for invcov_path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / invcov_path.format(collider=collider))
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
            '{collider}_tt_365': 'invcov_{collider}_tt_365GeV.dat'
        }
        self.n_dat = len(oo_tt_wc_basis)

    def compute_chi2(self, coefficient_values):

        chi2_value = 0
        for invcov_path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / invcov_path.format(collider=collider))
            chi2_value += np.linalg.multi_dot([coefficient_values, self.project.T, invcov, self.project, coefficient_values])
        return chi2_value






