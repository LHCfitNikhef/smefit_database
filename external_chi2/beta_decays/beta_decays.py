"""External chi2 for superallowed beta decays — rgevolve matching variant (Option 4).

At initialisation the derivative dL/dc_i of the LEC VnueduLL_1111 with respect to
each *free* SMEFT parameter is computed once analytically via rgevolve's run_and_match
matrix. Constrained operators (e.g. O3pl1=O3pl2=O3pl3=O3pl) are expanded automatically
via each Coefficient's vars/expr fields so their Warsaw-basis contributions are accumulated
into the corresponding free parameter's Jacobian entry. During sampling the LEC is
approximated

    L(c) ≈ L_SM + (dL/dc) · c_free

making compute_chi2 a pure JAX function, fully JIT-compatible with both BlackJAX and
UltraNest.

Compared to BetaDecayChi2Linear, this variant replaces the Wilson finite-difference
Jacobian with an analytic row vector extracted directly from rgevolve's precomputed
evolution and matching matrices. No Wilson calls are made at any stage — neither at
initialisation nor during sampling. The rgevolve matrices are cached by lru_cache, so
repeated instantiations (e.g. in multiprocessing) incur no re-computation cost.

The linear approximation and the NP-only L_SM=0 convention are identical to
BetaDecayChi2Linear; only the source of the Jacobian differs. Numerical differences
relative to Wilson are at the sub-percent level (rgevolve uses a different
implementation of the SMEFT→WET matching).

Note: importing this module triggers `import importlib.resources` before loading
rgevolve. This works around a Python 3.14 incompatibility in rgevolve's utils.py where
`importlib.resources` is accessed without being explicitly imported.

Example runcard entry::

    external_chi2:
      BetaDecayChi2RGevolve:
        path: /abs/path/to/external_chi2_beta_decay_rgevolve_matching.py
"""

from __future__ import annotations

import importlib.resources  # must precede rgevolve imports — Python 3.14 workaround

import warnings

import numpy as np
import jax
import jax.numpy as jnp
from smefit.rge import RGE
from rgevolve.tools.functions import run_and_match, get_wc_basis

jax.config.update("jax_enable_x64", True)

# Experimental Ft values and 1-sigma uncertainties from 2010.13797 (in units of 10^-3 s)
_EXP_MEAN = jnp.array(
    [
        3075.7,
        3070.2,
        3076.2,
        3072.4,
        3075.4,
        3071.6,
        3075.1,
        3072.9,
        3077.8,
        3071.7,
        3074.3,
        3071.1,
        3070.4,
        3072.4,
        3077.0,
    ]
)
_EXP_STD = jnp.array(
    [4.4, 1.9, 7.0, 1.1, 5.7, 1.8, 3.1, 2.0, 6.2, 2.0, 2.0, 1.6, 2.5, 6.7, 11.0]
)
# Q-values in MeV from https://journals.aps.org/prc/pdf/10.1103/PhysRevC.91.025501
_Q = jnp.array(
    [
        1.908,
        2.831,
        4.125,
        4.233,
        4.841,
        5.492,
        6.062,
        6.044,
        6.612,
        6.426,
        7.052,
        7.634,
        8.244,
        9.181,
        10.417,
    ]
)

_CONV = 1.52e24
_PREF = 4 * jnp.pi**3 * jnp.log(2.0) / (2 * (0.511e-3) ** 5)  # GeV^-1
_GF = 1.1663787e-5  # GeV^-2

_BD_PARAM_NAMES = {"DRV", "eta2", "eta3", "Vud"}


@jax.jit
def _chi2_sm(DRV, eta2, eta3, Vud):
    mean = _EXP_MEAN * _CONV
    std = _EXP_STD * _CONV
    Q = _Q * _CONV
    CV = jnp.sqrt(2.0) * _GF * Vud * jnp.sqrt(1.0 + DRV)
    Ft = _PREF / CV**2
    Ftt = Ft - eta2 * 3.3e-4 - eta3 * 8.0e-5 * Q
    return jnp.sum((Ftt - mean) ** 2 / std**2)


@jax.jit
def _chi2_smeft(DRV, eta2, eta3, Vud, L):
    mean = _EXP_MEAN * _CONV
    std = _EXP_STD * _CONV
    Q = _Q * _CONV
    Lf = -2.0 * jnp.sqrt(2.0) * _GF + L
    CV = -0.5 * Vud * Lf * jnp.sqrt(1.0 + DRV)
    Ft = _PREF / CV**2
    Ftt = Ft - eta2 * 3.3e-4 - eta3 * 8.0e-5 * Q
    return jnp.sum((Ftt - mean) ** 2 / std**2)


class BetaDecayChi2RGevolve:
    """SMEFiT external chi2 for superallowed beta decays — rgevolve Jacobian.

    The Jacobian dL/dc_i is computed analytically at initialisation via
    rgevolve.tools.functions.run_and_match, with no Wilson calls at any stage.
    The chi2 is then a pure JAX linear function of the free parameters, identical
    in form to BetaDecayChi2Linear but using rgevolve-derived derivatives.
    """

    def __init__(self, coefficients, rge_dict=None, starting_scale=None):
        free_names = list(coefficients.free_names)

        self._bd_idx = {
            n: free_names.index(n) for n in free_names if n in _BD_PARAM_NAMES
        }

        self._free_smeft_names = [n for n in free_names if n not in _BD_PARAM_NAMES]
        self._smeft_idx = jnp.array(
            [free_names.index(n) for n in self._free_smeft_names], dtype=jnp.int32
        )

        if starting_scale is not None:
            self._scale = float(starting_scale)
        elif rge_dict is not None:
            self._scale = float(rge_dict.get("init_scale", 10000.0))
        else:
            self._scale = 10000.0

        self.num_data = len(_EXP_MEAN)

        if not self._free_smeft_names:
            self._dL = None
            return

        smeft_accuracy = (
            rge_dict.get("smeft_accuracy", "integrate") if rge_dict else "integrate"
        )
        adm_QCD = rge_dict.get("adm_QCD", False) if rge_dict else False
        yukawa = rge_dict.get("yukawa", "top") if rge_dict else "top"

        all_smeft_names = [n for n in coefficients.names if n not in _BD_PARAM_NAMES]
        translation = RGE(
            all_smeft_names, self._scale, smeft_accuracy, adm_QCD, yukawa
        ).RGEbasis

        self._eff_translation = {n: {} for n in self._free_smeft_names}

        for coeff in coefficients.coefficients:
            if coeff.name in _BD_PARAM_NAMES:
                continue
            contrib = translation.get(coeff.name, {})
            if not contrib:
                continue

            if coeff.free:
                if coeff.name in self._eff_translation:
                    for k, v in contrib.items():
                        self._eff_translation[coeff.name][k] = (
                            self._eff_translation[coeff.name].get(k, 0.0) + v
                        )
            elif coeff.expr is not None and coeff.vars:
                for var in coeff.vars:
                    if var not in self._free_smeft_names:
                        continue
                    local = {v: (1.0 if v == var else 0.0) for v in coeff.vars}
                    try:
                        factor = float(eval(coeff.expr, {"__builtins__": {}}, local))
                    except Exception:
                        factor = 1.0
                    for k, v in contrib.items():
                        self._eff_translation[var][k] = (
                            self._eff_translation[var].get(k, 0.0) + factor * v
                        )

        self._dL = self._compute_jacobian_rgevolve()

    def _compute_jacobian_rgevolve(self):
        """Compute dL/dc_i analytically from the rgevolve run_and_match matrix.

        Collects all Warsaw WC names referenced in _eff_translation, queries rgevolve
        for the row vector d(VnueduLL_1111)/d(warsaw_wc_j), then contracts with the
        effective translation factors to get dL[i] = sum_j M[0,j] * eff[param_i][j].
        """
        # Collect Warsaw WC names that appear in any eff_translation entry.
        warsaw_names_needed: set[str] = set()
        for wc_dict in self._eff_translation.values():
            warsaw_names_needed.update(wc_dict.keys())

        if not warsaw_names_needed:
            return jnp.zeros(len(self._free_smeft_names))

        # Filter to names that exist as real WCs in rgevolve's Warsaw basis.
        smeft_warsaw_wcs = set(wc[0] for wc in get_wc_basis("SMEFT", "Warsaw"))
        unknown = warsaw_names_needed - smeft_warsaw_wcs
        if unknown:
            warnings.warn(
                f"BetaDecayChi2RGevolve: the following Warsaw WC names from the RGE "
                f"translation are not present in rgevolve's SMEFT Warsaw basis and will "
                f"be ignored: {sorted(unknown)}",
                stacklevel=2,
            )
        warsaw_list = sorted(warsaw_names_needed & smeft_warsaw_wcs)

        if not warsaw_list:
            return jnp.zeros(len(self._free_smeft_names))

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

        # Contract: dL[i] = sum_j m_row[j] * eff_translation[param_i].get(warsaw_j, 0)
        n = len(self._free_smeft_names)
        dL = np.zeros(n)
        for i, name in enumerate(self._free_smeft_names):
            for j, wc in enumerate(warsaw_list):
                factor = self._eff_translation[name].get(wc, 0.0)
                if factor != 0.0:
                    dL[i] += m_row[j] * factor

        return jnp.array(dL)

    def compute_chi2(self, coefficient_values):
        coefficient_values = jnp.asarray(coefficient_values)

        def _get(name, default):
            idx = self._bd_idx.get(name)
            return coefficient_values[idx] if idx is not None else default

        DRV = _get("DRV", 0.02467)
        eta2 = _get("eta2", 0.0)
        eta3 = _get("eta3", 0.0)
        Vud = _get("Vud", 0.9737)

        if self._dL is None:
            return _chi2_sm(DRV, eta2, eta3, Vud)

        smeft_vals = coefficient_values[self._smeft_idx]
        L = jnp.dot(self._dL, smeft_vals)
        return _chi2_smeft(DRV, eta2, eta3, Vud, L)
