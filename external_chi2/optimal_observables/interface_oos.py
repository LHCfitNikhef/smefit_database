import numpy as np
import pathlib
import jax.numpy as jnp

current_file_path = pathlib.Path(__file__).resolve().parent

# future colliders to include
collider = "FCCee"


class OptimalWW161:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "{collider}_ww_lepto_161": "invcov_{collider}_ww_leptonic_161.dat",
            "{collider}_ww_semilep_161": "invcov_{collider}_ww_semilep_161.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path.format(collider=collider))
            temp = jnp.einsum("ij, jk, kl", self.project.T, invcov, self.project)
            incovs_reordered.append(temp)
        self.incov_tot = jnp.sum(jnp.array(incovs_reordered), axis=0)

        self.rgemat = rgemat

        if self.rgemat is not None:
            # multiply the RGE matrix as well
            self.incov_tot = jnp.einsum(
                "ij, jk, kl", self.rgemat.T, self.incov_tot, self.rgemat
            )

        self.n_dat = len(oo_wc_basis)

    def compute_chi2(self, coefficient_values):
        chi2_value = jnp.einsum(
            "i, ij, j", coefficient_values, self.incov_tot, coefficient_values
        )

        return chi2_value


class OptimalWW240:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "{collider}_ww_lepto_240": "invcov_{collider}_ww_leptonic_240.dat",
            "{collider}_ww_semilep_240": "invcov_{collider}_ww_semilep_240.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path.format(collider=collider))
            temp = jnp.einsum("ij, jk, kl", self.project.T, invcov, self.project)
            incovs_reordered.append(temp)
        self.incov_tot = jnp.sum(jnp.array(incovs_reordered), axis=0)

        self.rgemat = rgemat

        if self.rgemat is not None:
            # multiply the RGE matrix as well
            self.incov_tot = jnp.einsum(
                "ij, jk, kl", self.rgemat.T, self.incov_tot, self.rgemat
            )

        self.n_dat = len(oo_wc_basis)

    def compute_chi2(self, coefficient_values):
        chi2_value = jnp.einsum(
            "i, ij, j", coefficient_values, self.incov_tot, coefficient_values
        )

        return chi2_value


class OptimalWW365:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "{collider}_ww_lepto_365": "invcov_{collider}_ww_leptonic_365.dat",
            "{collider}_ww_semilep_365": "invcov_{collider}_ww_semilep_365.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path.format(collider=collider))
            temp = jnp.einsum("ij, jk, kl", self.project.T, invcov, self.project)
            incovs_reordered.append(temp)
        self.incov_tot = jnp.sum(jnp.array(incovs_reordered), axis=0)

        self.rgemat = rgemat

        if self.rgemat is not None:
            # multiply the RGE matrix as well
            self.incov_tot = jnp.einsum(
                "ij, jk, kl", self.rgemat.T, self.incov_tot, self.rgemat
            )

        self.n_dat = len(oo_wc_basis)

    def compute_chi2(self, coefficient_values):
        chi2_value = jnp.einsum(
            "i, ij, j", coefficient_values, self.incov_tot, coefficient_values
        )

        return chi2_value


class OptimalWW:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "{collider}_ww_lepto_161": "invcov_{collider}_ww_leptonic_161.dat",
            "{collider}_ww_lepto_240": "invcov_{collider}_ww_leptonic_240.dat",
            "{collider}_ww_lepto_365": "invcov_{collider}_ww_leptonic_365.dat",
            "{collider}_ww_semilep_161": "invcov_{collider}_ww_semilep_161.dat",
            "{collider}_ww_semilep_240": "invcov_{collider}_ww_semilep_240.dat",
            "{collider}_ww_semilep_365": "invcov_{collider}_ww_semilep_365.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path.format(collider=collider))
            temp = jnp.einsum("ij, jk, kl", self.project.T, invcov, self.project)
            incovs_reordered.append(temp)
        self.incov_tot = jnp.sum(jnp.array(incovs_reordered), axis=0)

        self.rgemat = rgemat

        if self.rgemat is not None:
            # multiply the RGE matrix as well
            self.incov_tot = jnp.einsum(
                "ij, jk, kl", self.rgemat.T, self.incov_tot, self.rgemat
            )

        self.n_dat = len(oo_wc_basis)

    def compute_chi2(self, coefficient_values):
        chi2_value = jnp.einsum(
            "i, ij, j", coefficient_values, self.incov_tot, coefficient_values
        )

        return chi2_value


class OptimalttFCC365:
    def __init__(self, coefficients, rgemat=None):
        oo_tt_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "OQlM1", "OQe", "Otl1", "Ote"]

        self.project = np.zeros((len(oo_tt_wc_basis), coefficients.size))
        for i, op in enumerate(oo_tt_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        collider = "FCCee"
        self.datasets = {"{collider}_tt_365": "invcov_{collider}_tt_365GeV.dat"}

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path.format(collider=collider))
            temp = jnp.einsum("ij, jk, kl", self.project.T, invcov, self.project)
            incovs_reordered.append(temp)
        self.incov_tot = jnp.sum(jnp.array(incovs_reordered), axis=0)

        self.rgemat = rgemat

        if self.rgemat is not None:
            # multiply the RGE matrix as well
            self.incov_tot = jnp.einsum(
                "ij, jk, kl", self.rgemat.T, self.incov_tot, self.rgemat
            )

        self.n_dat = len(oo_tt_wc_basis)

    def compute_chi2(self, coefficient_values):
        chi2_value = jnp.einsum(
            "i, ij, j", coefficient_values, self.incov_tot, coefficient_values
        )

        return chi2_value
