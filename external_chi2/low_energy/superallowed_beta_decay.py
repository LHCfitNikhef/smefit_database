"""External chi2 for superallowed beta decays.

At initialisation the derivative dL/dc_i of the LEC VnueduLL_1111 with respect to
each SMEFT coefficient is computed once analytically via rgevolve's run_and_match
matrix:

    L(c) ~= L_SM + (dL/dc) . c

Example runcard entry::

    external_chi2:
      SA_beta_decays:
        path: smefit_database/external_chi2/low_energy/superallowed_beta_decay.py
"""

import importlib.resources  # must precede rgevolve imports — Python 3.14 workaround

import jax
import jax.numpy as jnp
import numpy as np
from rgevolve.tools.functions import get_wc_basis, run_and_match

from smefit import log
from smefit.rge.rge import RGE

_logger = log.logging.getLogger(__name__)

# Experimental Ft values (10^-3 s, 2010.13797), uncertainties, and Q-values (MeV,
# https://journals.aps.org/prc/pdf/10.1103/PhysRevC.91.025501) per nucleus
_NUCLEI = {
    "10C": {"mean": 3075.7, "std": 4.4, "Q": 1.908, "delta_r": 8.999999999999999e-05},
    "14O": {"mean": 3070.2, "std": 1.9, "Q": 2.831, "delta_r": 7.666666666666667e-05},
    "22Mg": {"mean": 3076.2, "std": 7.0, "Q": 4.125, "delta_r": 6.666666666666667e-05},
    "26Al": {"mean": 3072.4, "std": 1.1, "Q": 4.233, "delta_r": 6.666666666666667e-05},
    "26Si": {"mean": 3075.4, "std": 5.7, "Q": 4.841, "delta_r": 6.333333333333333e-05},
    "34Cl": {"mean": 3071.6, "std": 1.8, "Q": 5.492, "delta_r": 5.9999999999999995e-05},
    "34Ar": {"mean": 3075.1, "std": 3.1, "Q": 6.062, "delta_r": 5.666666666666667e-05},
    "38K": {"mean": 3072.9, "std": 2.0, "Q": 6.044, "delta_r": 5.666666666666667e-05},
    "38Ca": {"mean": 3077.8, "std": 6.2, "Q": 6.612, "delta_r": 5.666666666666667e-05},
    "42Sc": {"mean": 3071.7, "std": 2.0, "Q": 6.426, "delta_r": 5.666666666666667e-05},
    "46V": {"mean": 3074.3, "std": 2.0, "Q": 7.052, "delta_r": 5.333333333333333e-05},
    "50Mn": {"mean": 3071.1, "std": 1.6, "Q": 7.634, "delta_r": 5e-05},
    "54Co": {"mean": 3070.4, "std": 2.5, "Q": 8.244, "delta_r": 5e-05},
    "62Ga": {"mean": 3072.4, "std": 6.7, "Q": 9.181, "delta_r": 4.666666666666667e-05},
    "74Rb": {
        "mean": 3077.0,
        "std": 11.0,
        "Q": 10.417,
        "delta_r": 4.3333333333333334e-05,
    },
}

_EXP_MEAN = jnp.array([v["mean"] for v in _NUCLEI.values()])
_EXP_STD = jnp.array([v["std"] for v in _NUCLEI.values()])
_Q = jnp.array([v["Q"] for v in _NUCLEI.values()])
_DELTA_R = jnp.array([v["delta_r"] for v in _NUCLEI.values()])

_CONV = 1.519267e24
_PREF = 4 * jnp.pi**3 * jnp.log(2.0) / (2 * (0.5109989e-3) ** 5)  # GeV^-1
_GF = 1.16637859e-5  # GeV^-2

# Default SMEFT scale used when the runcard has no rge block and no
# starting_scale override.
_DEFAULT_SCALE = 10000.0

# Nuisance parameters of the beta-decay likelihood. They are ordinary runcard
# coefficients but carry no SMEFT operator, so they are excluded from the RGE
# translation (which would otherwise warn about them being unknown WCs).
_BD_PARAM_DEFAULTS = {
    "DRV": 0.02467,
    "eta1": 0.0,
    "eta2": 0.0,
    "eta3": 0.0,
    "Vud": 0.9737,
}


@jax.jit
def _chi2_smeft(DRV, eta1, eta2, eta3, Vud, L):
    """Beta-decay chi2 for a given LEC shift L.

    L = 0 reproduces the SM expression exactly, so no separate SM branch is
    needed: a runcard with no SMEFT coefficient simply yields a null Jacobian.
    """
    mean = _EXP_MEAN * _CONV
    std = _EXP_STD * _CONV
    Q = _Q
    Lf = -2.0 * jnp.sqrt(2.0) * _GF + L
    CV = -0.5 * Vud * Lf * jnp.sqrt(1.0 + DRV)
    Ft = _PREF / CV**2
    Ftt = Ft - mean * (eta1 * _DELTA_R + eta2 * 3.3e-4 + eta3 * 8.0e-5 * Q)
    return jnp.sum((Ftt - mean) ** 2 / std**2)


class SA_beta_decays:
    """SMEFiT external chi2 for superallowed beta decays — rgevolve Jacobian.

    The Jacobian dL/dc is computed analytically at initialisation via
    rgevolve.tools.functions.run_and_match, with no Wilson calls at any stage.
    The chi2 is then a pure JAX function of the full coefficient vector.
    """

    def __init__(self, coefficients, rge_dict=None, starting_scale=None):
        """
        Initialize the SA_beta_decays class.

        coefficients: The Wilson coefficients to be used in the analysis.
        rge_dict: A dictionary containing the RGE information.
        starting_scale: SMEFT scale (GeV) at which the coefficients are defined.
            Overrides rge_dict["init_scale"] when given.
        """
        # cast away numpy.str_ so plain string lookups stay exact
        self.coeff_names = [str(n) for n in coefficients.name]

        # Indices into the full coefficient vector passed to compute_chi2.
        self._bd_idx = {
            name: self.coeff_names.index(name)
            for name in _BD_PARAM_DEFAULTS
            if name in self.coeff_names
        }

        if starting_scale is not None:
            self._scale = float(starting_scale)
        elif rge_dict is not None:
            self._scale = float(rge_dict.get("init_scale", _DEFAULT_SCALE))
        else:
            self._scale = _DEFAULT_SCALE

        self.n_dat = len(_EXP_MEAN)

        # The smefit -> Warsaw translation comes from the standard smefit
        # runner; only the SMEFT -> WET matching below is rgevolve's.
        smeft_names = [n for n in self.coeff_names if n not in _BD_PARAM_DEFAULTS]
        if smeft_names:
            translation = RGE(
                wc_names=smeft_names,
                init_scale=self._scale,
                accuracy=(
                    rge_dict.get("smeft_accuracy", "integrate")
                    if rge_dict
                    else "integrate"
                ),
                adm_QCD=rge_dict.get("adm_QCD", False) if rge_dict else False,
                yukawa=rge_dict.get("yukawa", "top") if rge_dict else "top",
            ).RGEbasis
        else:
            translation = {}

        self._dL = self._compute_jacobian_rgevolve(translation)

    def _compute_jacobian_rgevolve(self, translation):
        """Compute dL/dc analytically from the rgevolve run_and_match matrix.

        Collects all Warsaw WC names referenced by the smefit -> Warsaw
        translation, queries rgevolve for the row vector
        d(VnueduLL_1111)/d(warsaw_wc_j), then contracts it with the translation
        factors.

        Returns a jnp array of shape (len(coefficients.name),), aligned with the
        full coefficient vector. Entries for the beta-decay nuisance parameters
        are zero.
        """
        n_coeff = len(self.coeff_names)

        warsaw_names_needed = set()
        for wc_dict in translation.values():
            warsaw_names_needed.update(wc_dict.keys())

        if not warsaw_names_needed:
            return jnp.zeros(n_coeff)

        # Filter to names that exist as real WCs in rgevolve's Warsaw basis.
        smeft_warsaw_wcs = {wc[0] for wc in get_wc_basis("SMEFT", "Warsaw")}
        unknown = warsaw_names_needed - smeft_warsaw_wcs
        if unknown:
            _logger.warning(
                "SA_beta_decays: the following Warsaw WC names from the RGE "
                "translation are not present in rgevolve's SMEFT Warsaw basis "
                "and will be ignored: %s",
                sorted(unknown),
            )
        warsaw_list = sorted(warsaw_names_needed & smeft_warsaw_wcs)

        if not warsaw_list:
            return jnp.zeros(n_coeff)

        wcs_in = tuple((name, "R") for name in warsaw_list)
        wcs_out = (("VnueduLL_1111", "R"),)

        # run_and_match returns shape (len(wcs_out), len(wcs_in)) = (1, N).
        # M[0, j] = d(VnueduLL_1111_R) / d(warsaw_list[j]_R) at the given scales.
        M = run_and_match(
            "SMEFT",
            "WET",
            "Warsaw",
            "JMS",
            self._scale,
            2.0,
            wcs_in=wcs_in,
            wcs_out=wcs_out,
        )
        m_row = M[0]  # shape (N,)

        dL = np.zeros(n_coeff)
        for name, wc_dict in translation.items():
            i = self.coeff_names.index(name)
            for j, wc in enumerate(warsaw_list):
                factor = wc_dict.get(wc, 0.0)
                if factor != 0.0:
                    dL[i] += m_row[j] * factor

        return jnp.array(dL)

    def compute_chi2(self, coefficient_values):
        # coefficient_values is the full coefficient vector: the optimizer has
        # already placed the free parameters and applied the constraints.
        def _get(name):
            idx = self._bd_idx.get(name)
            if idx is None:
                return jnp.asarray(_BD_PARAM_DEFAULTS[name])
            return coefficient_values[idx]

        L = jnp.dot(self._dL, coefficient_values)

        return _chi2_smeft(
            _get("DRV"),
            _get("eta1"),
            _get("eta2"),
            _get("eta3"),
            _get("Vud"),
            L,
        )
