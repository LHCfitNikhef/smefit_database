import numpy as np
import pathlib
import jax.numpy as jnp

current_file_path = pathlib.Path(__file__).resolve().parent

# future colliders to include
collider = 'FCCee'


class OptimalWW:

    def __init__(self, coefficients):

        oo_wc_basis = ['OpD', 'OpWB', 'OWWW', 'Opl1', 'Ope', 'O3pl1']

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {

            '{collider}_ww_lepto_240': 'invcov_{collider}_ww_leptonic_240.dat',
            '{collider}_ww_semilep_240': 'invcov_{collider}_ww_semilep_240.dat',

        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path.format(collider=collider))
            temp = self.project.T @ invcov @ self.project
            incovs_reordered.append(temp)

        self.incovs_reordered = jnp.sum(jnp.array(incovs_reordered), axis=0)

        self.n_dat = len(oo_wc_basis)

    def compute_chi2(self, coefficient_values):

        chi2_value = coefficient_values @ self.incovs_reordered @ coefficient_values

        return chi2_value








