import numpy as np
import pathlib
import jax.numpy as jnp

current_file_path = pathlib.Path(__file__).resolve().parent

# future colliders to include
collider = "LCF"


class OptimalWWLCF250:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "LCF_ww_lepto_250_neg80_neg30": "invcov_LCF_ww_leptonic_250_neg80_neg30.dat",
            "LCF_ww_lepto_250_neg80_pos30": "invcov_LCF_ww_leptonic_250_neg80_pos30.dat",
            "LCF_ww_lepto_250_pos80_neg30": "invcov_LCF_ww_leptonic_250_pos80_neg30.dat",
            "LCF_ww_lepto_250_pos80_pos30": "invcov_LCF_ww_leptonic_250_pos80_pos30.dat",
            "LCF_ww_semilep_250_neg80_neg30": "invcov_LCF_ww_semilep_250_neg80_neg30.dat",
            "LCF_ww_semilep_250_neg80_pos30": "invcov_LCF_ww_semilep_250_neg80_pos30.dat",
            "LCF_ww_semilep_250_pos80_neg30": "invcov_LCF_ww_semilep_250_pos80_neg30.dat",
            "LCF_ww_semilep_250_pos80_pos30": "invcov_LCF_ww_semilep_250_pos80_pos30.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path)
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


class OptimalWWLCF5004ab:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "LCF_ww_lepto_500_4ab_neg80_neg30": "invcov_LCF_ww_leptonic_500_4ab_neg80_neg30.dat",
            "LCF_ww_lepto_500_4ab_neg80_pos30": "invcov_LCF_ww_leptonic_500_4ab_neg80_pos30.dat",
            "LCF_ww_lepto_500_4ab_pos80_neg30": "invcov_LCF_ww_leptonic_500_4ab_pos80_neg30.dat",
            "LCF_ww_lepto_500_4ab_pos80_pos30": "invcov_LCF_ww_leptonic_500_4ab_pos80_pos30.dat",
            "LCF_ww_semilep_500_4ab_neg80_neg30": "invcov_LCF_ww_semilep_500_4ab_neg80_neg30.dat",
            "LCF_ww_semilep_500_4ab_neg80_pos30": "invcov_LCF_ww_semilep_500_4ab_neg80_pos30.dat",
            "LCF_ww_semilep_500_4ab_pos80_neg30": "invcov_LCF_ww_semilep_500_4ab_pos80_neg30.dat",
            "LCF_ww_semilep_500_4ab_pos80_pos30": "invcov_LCF_ww_semilep_500_4ab_pos80_pos30.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path)
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

class OptimalWWLCF5008ab:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "LCF_ww_lepto_500_8ab_neg80_neg30": "invcov_LCF_ww_leptonic_500_8ab_neg80_neg30.dat",
            "LCF_ww_lepto_500_8ab_neg80_pos30": "invcov_LCF_ww_leptonic_500_8ab_neg80_pos30.dat",
            "LCF_ww_lepto_500_8ab_pos80_neg30": "invcov_LCF_ww_leptonic_500_8ab_pos80_neg30.dat",
            "LCF_ww_lepto_500_8ab_pos80_pos30": "invcov_LCF_ww_leptonic_500_8ab_pos80_pos30.dat",
            "LCF_ww_semilep_500_8ab_neg80_neg30": "invcov_LCF_ww_semilep_500_8ab_neg80_neg30.dat",
            "LCF_ww_semilep_500_8ab_neg80_pos30": "invcov_LCF_ww_semilep_500_8ab_neg80_pos30.dat",
            "LCF_ww_semilep_500_8ab_pos80_neg30": "invcov_LCF_ww_semilep_500_8ab_pos80_neg30.dat",
            "LCF_ww_semilep_500_8ab_pos80_pos30": "invcov_LCF_ww_semilep_500_8ab_pos80_pos30.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path)
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


class OptimalWWLCF1000:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "LCF_ww_lepto_1000_neg80_neg20": "invcov_LCF_ww_leptonic_1000_neg80_neg20.dat",
            "LCF_ww_lepto_1000_neg80_pos20": "invcov_LCF_ww_leptonic_1000_neg80_pos20.dat",
            "LCF_ww_lepto_1000_pos80_neg20": "invcov_LCF_ww_leptonic_1000_pos80_neg20.dat",
            "LCF_ww_lepto_1000_pos80_pos20": "invcov_LCF_ww_leptonic_1000_pos80_pos20.dat",
            "LCF_ww_semilep_1000_neg80_neg20": "invcov_LCF_ww_semilep_1000_neg80_neg20.dat",
            "LCF_ww_semilep_1000_neg80_pos20": "invcov_LCF_ww_semilep_1000_neg80_pos20.dat",
            "LCF_ww_semilep_1000_pos80_neg20": "invcov_LCF_ww_semilep_1000_pos80_neg20.dat",
            "LCF_ww_semilep_1000_pos80_pos20": "invcov_LCF_ww_semilep_1000_pos80_pos20.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path)
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
        
class OptimalttLCF350:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "LCF_tt_wbwb_350_neg80_neg30": "invcov_LCF_tt_wbwb_350_neg80_neg30.dat",
            "LCF_tt_wbwb_350_pos80_neg30": "invcov_LCF_tt_wbwb_350_pos80_neg30.dat",
            "LCF_tt_wbwb_350_neg80_pos30": "invcov_LCF_tt_wbwb_350_neg80_pos30.dat",
            "LCF_tt_wbwb_350_pos80_pos30": "invcov_LCF_tt_wbwb_350_pos80_pos30.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path)
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


class OptimalttLCF350full:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "Ol1QM", "OeQ", "Ol1t", "Oet"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "LCF_tt_wbwb_350_neg80_neg30": "invcov_LCF_tt_wbwb_350_neg80_neg30_full.dat",
            "LCF_tt_wbwb_350_pos80_neg30": "invcov_LCF_tt_wbwb_350_pos80_neg30_full.dat",
            "LCF_tt_wbwb_350_neg80_pos30": "invcov_LCF_tt_wbwb_350_neg80_pos30_full.dat",
            "LCF_tt_wbwb_350_pos80_pos30": "invcov_LCF_tt_wbwb_350_pos80_pos30_full.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path)
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


class OptimalttLCF5004ab:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "LCF_tt_wbwb_500_4ab_neg80_neg30": "invcov_LCF_tt_wbwb_500_4ab_neg80_neg30.dat",
            "LCF_tt_wbwb_500_4ab_pos80_neg30": "invcov_LCF_tt_wbwb_500_4ab_pos80_neg30.dat",
            "LCF_tt_wbwb_500_4ab_neg80_pos30": "invcov_LCF_tt_wbwb_500_4ab_neg80_pos30.dat",
            "LCF_tt_wbwb_500_4ab_pos80_pos30": "invcov_LCF_tt_wbwb_500_4ab_pos80_pos30.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path)
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

class OptimalttLCF5008ab:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "LCF_tt_wbwb_500_8ab_neg80_neg30": "invcov_LCF_tt_wbwb_500_8ab_neg80_neg30.dat",
            "LCF_tt_wbwb_500_8ab_pos80_neg30": "invcov_LCF_tt_wbwb_500_8ab_pos80_neg30.dat",
            "LCF_tt_wbwb_500_8ab_neg80_pos30": "invcov_LCF_tt_wbwb_500_8ab_neg80_pos30.dat",
            "LCF_tt_wbwb_500_8ab_pos80_pos30": "invcov_LCF_tt_wbwb_500_8ab_pos80_pos30.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path)
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

class OptimalttLCF5004abfull:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "Ol1QM", "OeQ", "Ol1t", "Oet"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "LCF_tt_wbwb_500_4ab_neg80_neg30": "invcov_LCF_tt_wbwb_500_4ab_neg80_neg30_full.dat",
            "LCF_tt_wbwb_500_4ab_pos80_neg30": "invcov_LCF_tt_wbwb_500_4ab_pos80_neg30_full.dat",
            "LCF_tt_wbwb_500_4ab_neg80_pos30": "invcov_LCF_tt_wbwb_500_4ab_neg80_pos30_full.dat",
            "LCF_tt_wbwb_500_4ab_pos80_pos30": "invcov_LCF_tt_wbwb_500_4ab_pos80_pos30_full.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path)
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
        
class OptimalttLCF5008abfull:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "Ol1QM", "OeQ", "Ol1t", "Oet"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "LCF_tt_wbwb_500_8ab_neg80_neg30": "invcov_LCF_tt_wbwb_500_8ab_neg80_neg30_full.dat",
            "LCF_tt_wbwb_500_8ab_pos80_neg30": "invcov_LCF_tt_wbwb_500_8ab_pos80_neg30_full.dat",
            "LCF_tt_wbwb_500_8ab_neg80_pos30": "invcov_LCF_tt_wbwb_500_8ab_neg80_pos30_full.dat",
            "LCF_tt_wbwb_500_8ab_pos80_pos30": "invcov_LCF_tt_wbwb_500_8ab_pos80_pos30_full.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path)
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


class OptimalttLCF1000:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "LCF_tt_wbwb_1000_neg80_neg20": "invcov_LCF_tt_wbwb_1000_neg80_neg20.dat",
            "LCF_tt_wbwb_1000_pos80_neg20": "invcov_LCF_tt_wbwb_1000_pos80_neg20.dat",
            "LCF_tt_wbwb_1000_neg80_pos20": "invcov_LCF_tt_wbwb_1000_neg80_pos20.dat",
            "LCF_tt_wbwb_1000_pos80_pos20": "invcov_LCF_tt_wbwb_1000_pos80_pos20.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path)
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


class OptimalttLCF1000full:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "Ol1QM", "OeQ", "Ol1t", "Oet"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "LCF_tt_wbwb_1000_neg80_neg20": "invcov_LCF_tt_wbwb_1000_neg80_neg20_full.dat",
            "LCF_tt_wbwb_1000_pos80_neg20": "invcov_LCF_tt_wbwb_1000_pos80_neg20_full.dat",
            "LCF_tt_wbwb_1000_neg80_pos20": "invcov_LCF_tt_wbwb_1000_neg80_pos20_full.dat",
            "LCF_tt_wbwb_1000_pos80_pos20": "invcov_LCF_tt_wbwb_1000_pos80_pos20_full.dat",
        }

        incovs_reordered = []
        for path in self.datasets.values():
            invcov = np.loadtxt(current_file_path / path)
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


