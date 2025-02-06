import numpy as np
import pathlib
import jax.numpy as jnp

current_file_path = pathlib.Path(__file__).resolve().parent


class OptimalWWFCC161:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        collider = "FCCee"
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


class OptimalWWFCC240:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        collider = "FCCee"
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


class OptimalWWFCC365:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        collider = "FCCee"
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


class OptimalWWCEPC161:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        collider = "CEPC"
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


class OptimalWWCEPC240:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        collider = "CEPC"
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


class OptimalWWCEPC365:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        collider = "CEPC"
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


class OptimalWWILC250:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "ILC_ww_lepto_250_neg80_neg30": "invcov_ILC_ww_leptonic_250_neg80_neg30.dat",
            "ILC_ww_lepto_250_neg80_pos30": "invcov_ILC_ww_leptonic_250_neg80_pos30.dat",
            "ILC_ww_lepto_250_pos80_neg30": "invcov_ILC_ww_leptonic_250_pos80_neg30.dat",
            "ILC_ww_lepto_250_pos80_pos30": "invcov_ILC_ww_leptonic_250_pos80_pos30.dat",
            "ILC_ww_semilep_250_neg80_neg30": "invcov_ILC_ww_semilep_250_neg80_neg30.dat",
            "ILC_ww_semilep_250_neg80_pos30": "invcov_ILC_ww_semilep_250_neg80_pos30.dat",
            "ILC_ww_semilep_250_pos80_neg30": "invcov_ILC_ww_semilep_250_pos80_neg30.dat",
            "ILC_ww_semilep_250_pos80_pos30": "invcov_ILC_ww_semilep_250_pos80_pos30.dat",
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


class OptimalWWILC500:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "ILC_ww_lepto_500_neg80_neg30": "invcov_ILC_ww_leptonic_500_neg80_neg30.dat",
            "ILC_ww_lepto_500_neg80_pos30": "invcov_ILC_ww_leptonic_500_neg80_pos30.dat",
            "ILC_ww_lepto_500_pos80_neg30": "invcov_ILC_ww_leptonic_500_pos80_neg30.dat",
            "ILC_ww_lepto_500_pos80_pos30": "invcov_ILC_ww_leptonic_500_pos80_pos30.dat",
            "ILC_ww_semilep_500_neg80_neg30": "invcov_ILC_ww_semilep_500_neg80_neg30.dat",
            "ILC_ww_semilep_500_neg80_pos30": "invcov_ILC_ww_semilep_500_neg80_pos30.dat",
            "ILC_ww_semilep_500_pos80_neg30": "invcov_ILC_ww_semilep_500_pos80_neg30.dat",
            "ILC_ww_semilep_500_pos80_pos30": "invcov_ILC_ww_semilep_500_pos80_pos30.dat",
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


class OptimalWWILC1000:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "ILC_ww_lepto_1000_neg80_neg20": "invcov_ILC_ww_leptonic_1000_neg80_neg20.dat",
            "ILC_ww_lepto_1000_neg80_pos20": "invcov_ILC_ww_leptonic_1000_neg80_pos20.dat",
            "ILC_ww_lepto_1000_pos80_neg20": "invcov_ILC_ww_leptonic_1000_pos80_neg20.dat",
            "ILC_ww_lepto_1000_pos80_pos20": "invcov_ILC_ww_leptonic_1000_pos80_pos20.dat",
            "ILC_ww_semilep_1000_neg80_neg20": "invcov_ILC_ww_semilep_1000_neg80_neg20.dat",
            "ILC_ww_semilep_1000_neg80_pos20": "invcov_ILC_ww_semilep_1000_neg80_pos20.dat",
            "ILC_ww_semilep_1000_pos80_neg20": "invcov_ILC_ww_semilep_1000_pos80_neg20.dat",
            "ILC_ww_semilep_1000_pos80_pos20": "invcov_ILC_ww_semilep_1000_pos80_pos20.dat",
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


class OptimalWWCLIC380:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "CLIC_ww_lepto_380_neg80_0": "invcov_CLIC_ww_leptonic_380_neg80_0.dat",
            "CLIC_ww_lepto_380_pos80_0": "invcov_CLIC_ww_leptonic_380_pos80_0.dat",
            "CLIC_ww_semilep_380_neg80_0": "invcov_CLIC_ww_semilep_380_neg80_0.dat",
            "CLIC_ww_semilep_380_pos80_0": "invcov_CLIC_ww_semilep_380_pos80_0.dat",
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


class OptimalWWCLIC1500:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "CLIC_ww_lepto_1500_neg80_0": "invcov_CLIC_ww_leptonic_1500_neg80_0.dat",
            "CLIC_ww_lepto_1500_pos80_0": "invcov_CLIC_ww_leptonic_1500_pos80_0.dat",
            "CLIC_ww_semilep_1500_neg80_0": "invcov_CLIC_ww_semilep_1500_neg80_0.dat",
            "CLIC_ww_semilep_1500_pos80_0": "invcov_CLIC_ww_semilep_1500_pos80_0.dat",
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


class OptimalWWCLIC3000:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "CLIC_ww_lepto_3000_neg80_0": "invcov_CLIC_ww_leptonic_3000_neg80_0.dat",
            "CLIC_ww_lepto_3000_pos80_0": "invcov_CLIC_ww_leptonic_3000_pos80_0.dat",
            "CLIC_ww_semilep_3000_neg80_0": "invcov_CLIC_ww_semilep_3000_neg80_0.dat",
            "CLIC_ww_semilep_3000_pos80_0": "invcov_CLIC_ww_semilep_3000_pos80_0.dat",
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


class OptimalttFCC365:
    def __init__(self, coefficients, rgemat=None):
        oo_tt_wc_basis = ["OpQM", "Opt", "OtW", "OtZ"]

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


class OptimalttFCC365full:
    def __init__(self, coefficients, rgemat=None):
        oo_tt_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "Ol1QM", "OeQ", "Ol1t", "Oet"]

        self.project = np.zeros((len(oo_tt_wc_basis), coefficients.size))
        for i, op in enumerate(oo_tt_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        collider = "FCCee"
        self.datasets = {"{collider}_tt_365": "invcov_{collider}_tt_365GeV_full.dat"}

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


class OptimalttCEPC365:
    def __init__(self, coefficients, rgemat=None):
        oo_tt_wc_basis = ["OpQM", "Opt", "OtW", "OtZ"]

        self.project = np.zeros((len(oo_tt_wc_basis), coefficients.size))
        for i, op in enumerate(oo_tt_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        collider = "CEPC"
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


class OptimalttCEPC365full:
    def __init__(self, coefficients, rgemat=None):
        oo_tt_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "Ol1QM", "OeQ", "Ol1t", "Oet"]

        self.project = np.zeros((len(oo_tt_wc_basis), coefficients.size))
        for i, op in enumerate(oo_tt_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        collider = "CEPC"
        self.datasets = {"{collider}_tt_365": "invcov_{collider}_tt_365GeV_full.dat"}

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


class OptimalttCLIC380:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "CLIC_tt_wbwb_380_neg80_0": "invcov_CLIC_tt_wbwb_380_neg80_0.dat",
            "CLIC_tt_wbwb_380_pos80_0": "invcov_CLIC_tt_wbwb_380_pos80_0.dat",
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


class OptimalttCLIC380full:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "Ol1QM", "OeQ", "Ol1t", "Oet"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "CLIC_tt_wbwb_380_neg80_0": "invcov_CLIC_tt_wbwb_380_neg80_0_full.dat",
            "CLIC_tt_wbwb_380_pos80_0": "invcov_CLIC_tt_wbwb_380_pos80_0_full.dat",
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


class OptimalttCLIC1500:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "CLIC_tt_wbwb_1500_neg80_0": "invcov_CLIC_tt_wbwb_1500_neg80_0.dat",
            "CLIC_tt_wbwb_1500_pos80_0": "invcov_CLIC_tt_wbwb_1500_pos80_0.dat",
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


class OptimalttCLIC1500full:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "Ol1QM", "OeQ", "Ol1t", "Oet"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "CLIC_tt_wbwb_1500_neg80_0": "invcov_CLIC_tt_wbwb_1500_neg80_0_full.dat",
            "CLIC_tt_wbwb_1500_pos80_0": "invcov_CLIC_tt_wbwb_1500_pos80_0_full.dat",
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


class OptimalttCLIC3000:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "CLIC_tt_wbwb_3000_neg80_0": "invcov_CLIC_tt_wbwb_3000_neg80_0.dat",
            "CLIC_tt_wbwb_3000_pos80_0": "invcov_CLIC_tt_wbwb_3000_pos80_0.dat",
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


class OptimalttCLIC3000full:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "Ol1QM", "OeQ", "Ol1t", "Oet"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "CLIC_tt_wbwb_3000_neg80_0": "invcov_CLIC_tt_wbwb_3000_neg80_0_full.dat",
            "CLIC_tt_wbwb_3000_pos80_0": "invcov_CLIC_tt_wbwb_3000_pos80_0_full.dat",
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


class OptimalttILC350:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "ILC_tt_wbwb_350_neg80_neg30": "invcov_ILC_tt_wbwb_350_neg80_neg30.dat",
            "ILC_tt_wbwb_350_pos80_neg30": "invcov_ILC_tt_wbwb_350_pos80_neg30.dat",
            "ILC_tt_wbwb_350_neg80_pos30": "invcov_ILC_tt_wbwb_350_neg80_pos30.dat",
            "ILC_tt_wbwb_350_pos80_pos30": "invcov_ILC_tt_wbwb_350_pos80_pos30.dat",
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


class OptimalttILC350full:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "Ol1QM", "OeQ", "Ol1t", "Oet"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "ILC_tt_wbwb_350_neg80_neg30": "invcov_ILC_tt_wbwb_350_neg80_neg30_full.dat",
            "ILC_tt_wbwb_350_pos80_neg30": "invcov_ILC_tt_wbwb_350_pos80_neg30_full.dat",
            "ILC_tt_wbwb_350_neg80_pos30": "invcov_ILC_tt_wbwb_350_neg80_pos30_full.dat",
            "ILC_tt_wbwb_350_pos80_pos30": "invcov_ILC_tt_wbwb_350_pos80_pos30_full.dat",
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


class OptimalttILC500:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "ILC_tt_wbwb_500_neg80_neg30": "invcov_ILC_tt_wbwb_500_neg80_neg30.dat",
            "ILC_tt_wbwb_500_pos80_neg30": "invcov_ILC_tt_wbwb_500_pos80_neg30.dat",
            "ILC_tt_wbwb_500_neg80_pos30": "invcov_ILC_tt_wbwb_500_neg80_pos30.dat",
            "ILC_tt_wbwb_500_pos80_pos30": "invcov_ILC_tt_wbwb_500_pos80_pos30.dat",
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


class OptimalttILC500full:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "Ol1QM", "OeQ", "Ol1t", "Oet"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "ILC_tt_wbwb_500_neg80_neg30": "invcov_ILC_tt_wbwb_500_neg80_neg30_full.dat",
            "ILC_tt_wbwb_500_pos80_neg30": "invcov_ILC_tt_wbwb_500_pos80_neg30_full.dat",
            "ILC_tt_wbwb_500_neg80_pos30": "invcov_ILC_tt_wbwb_500_neg80_pos30_full.dat",
            "ILC_tt_wbwb_500_pos80_pos30": "invcov_ILC_tt_wbwb_500_pos80_pos30_full.dat",
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


class OptimalttILC1000:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "ILC_tt_wbwb_1000_neg80_neg20": "invcov_ILC_tt_wbwb_1000_neg80_neg20.dat",
            "ILC_tt_wbwb_1000_pos80_neg20": "invcov_ILC_tt_wbwb_1000_pos80_neg20.dat",
            "ILC_tt_wbwb_1000_neg80_pos20": "invcov_ILC_tt_wbwb_1000_neg80_pos20.dat",
            "ILC_tt_wbwb_1000_pos80_pos20": "invcov_ILC_tt_wbwb_1000_pos80_pos20.dat",
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


class OptimalttILC1000full:
    def __init__(self, coefficients, rgemat=None):
        oo_wc_basis = ["OpQM", "Opt", "OtW", "OtZ", "Ol1QM", "OeQ", "Ol1t", "Oet"]

        self.project = np.zeros((len(oo_wc_basis), coefficients.size))
        for i, op in enumerate(oo_wc_basis):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        self.datasets = {
            "ILC_tt_wbwb_1000_neg80_neg20": "invcov_ILC_tt_wbwb_1000_neg80_neg20_full.dat",
            "ILC_tt_wbwb_1000_pos80_neg20": "invcov_ILC_tt_wbwb_1000_pos80_neg20_full.dat",
            "ILC_tt_wbwb_1000_neg80_pos20": "invcov_ILC_tt_wbwb_1000_neg80_pos20_full.dat",
            "ILC_tt_wbwb_1000_pos80_pos20": "invcov_ILC_tt_wbwb_1000_pos80_pos20_full.dat",
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
