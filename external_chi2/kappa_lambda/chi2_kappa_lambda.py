import jax.numpy as jnp
import numpy as np
from pathlib import Path
import pickle

# Get the directory of the current file
current_dir = Path(__file__).parent


class HLLHC_kappa_lambda_chi2:
    def __init__(self, coefficients, rgemat=None):
        self.wc = [
            "Op",
        ]
        self.rgemat = rgemat
        self.chi2_path = current_dir / "kappa_lambda_HL-LHC_likelihood.pkl"

        # coefficients is a list with the names of the coefficients. We need to create a projection matrix
        # that projects the coefficients to the basis of the observables
        self.project = np.zeros((len(self.wc), coefficients.size))
        for i, op in enumerate(self.wc):
            if op in coefficients.name:
                self.project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

        if self.rgemat is not None:
            self.project = jnp.einsum("ij,jk->ik", self.project, self.rgemat)

        # load pickle file
        with open(self.chi2_path, "rb") as f:
            self.chi2 = pickle.load(f)

    def compute_chi2(self, coefficient_values):
        coeffs_vals = jnp.dot(self.project, coefficient_values)
        # now we have to transform cphi in klambda
        klambda = 1 - 0.47 * coeffs_vals

        chi2_value = self.chi2(klambda).squeeze()

        return chi2_value
