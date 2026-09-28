"""External chi2 for leptonic meson decays (π/K) — rgevolve matching variant.

Computes chi2 from four observables:
  Rπ  = Γ(π→eν)/Γ(π→μν)
  RK  = Γ(K→eν)/Γ(K→μν)
  Rμ  = Γ(K→μν)/Γ(π→μν)
  Γ(K→μν)

SMEFT enters through four WET LEFT coefficients at 2 GeV (JMS basis):
  VnueduLL_1111  for π→eν   (labelled L1111 in chi2meson2.py)
  VnueduLL_2211  for π→μν   (labelled L2211)
  VnueduLL_1112  for K→eν   (labelled L1121 in chi2meson2.py; 4th index = 2 = s quark)
  VnueduLL_2212  for K→μν   (labelled L2221 in chi2meson2.py)

At initialisation, Jacobians dL_X/dc_i are computed analytically via rgevolve for each
of the four LEFT WCs. During sampling the LEFT coefficients are approximated as

    L_X(c) ≈ dot(dL_X, c_smeft)

making compute_chi2 a pure JAX function, fully JIT-compatible with both BlackJAX and
UltraNest. No Wilson or rgevolve calls are made at sampling time.

The FF function (from chi2meson2.py) contains mpmath.polylog which is not JAX-traceable.
All four FF evaluations are pre-computed as Python floats at module load time since their
arguments (me/mpi, mmu/mpi, me/mK, mmu/mK) are physical constants, not fit parameters.

Nuisance parameters that can be declared free in the runcard:
  c1p, fkfpi, fPK, Sew   (EM corrections, Gaussian-distributed per chi2meson2.Gpars)
  Vud, Vus               (CKM elements)

Note: importing this module triggers `import importlib.resources` before loading rgevolve,
working around a Python 3.14 incompatibility in rgevolve's utils.py.

Example runcard entry::

    external_chi2:
      MesonDecayChi2:
        path: /abs/path/to/external_chi2_meson_decays.py
"""

from __future__ import annotations

import importlib.resources  # must precede rgevolve imports — Python 3.14 workaround
import warnings

import mpmath
import numpy as np
import jax
import jax.numpy as jnp
from smefit.rge import RGE
from rgevolve.tools.functions import run_and_match, get_wc_basis

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Physical constants (not fit parameters)
# ---------------------------------------------------------------------------
_mpi = 0.13957039  # GeV
_mK = 0.493677  # GeV
_mmu = 0.10565837  # GeV
_me = 0.5109989e-3  # GeV
_GF = 1.1663787e-5  # GeV^-2
_mrho = 0.77526  # GeV
_alpha = 0.00729735


# ---------------------------------------------------------------------------
# Pre-compute FF values with mpmath (contains polylog, not JAX-traceable).
# Arguments are physical constants so these are module-level floats.
# FF(x) from chi2meson2.py:
#   3/2 log(x) + (13-19x)/(8(1-x)) - (8-5x)x log(x)/(4(1-x)^2)
#   - (2 + (1+x)/(1-x) log(x)) log(1-x) - 2(1+x)/(1-x) Li2(1-x)
# ---------------------------------------------------------------------------
def _FF_mpmath(x):
    x = mpmath.mpf(x)
    return (
        mpmath.mpf("3") / 2 * mpmath.log(x)
        + (13 - 19 * x) / (8 * (1 - x))
        - (8 - 5 * x) / (4 * (1 - x) ** 2) * x * mpmath.log(x)
        - (2 + (1 + x) / (1 - x) * mpmath.log(x)) * mpmath.log(1 - x)
        - 2 * (1 + x) / (1 - x) * mpmath.polylog(2, 1 - x)
    )


_FF_me_pi = float(_FF_mpmath(_me**2 / _mpi**2))
_FF_mmu_pi = float(_FF_mpmath(_mmu**2 / _mpi**2))
_FF_me_K = float(_FF_mpmath(_me**2 / _mK**2))
_FF_mmu_K = float(_FF_mpmath(_mmu**2 / _mK**2))

# ---------------------------------------------------------------------------
# Pre-compute other constants that appear in the EM corrections.
# Original formula:  delta = Sew*(1 + alpha/pi*FF(x) - alpha/pi*(-3/2*log(m/mrho) + c1)) - 1
#                          = Sew*(1 + alpha/pi*(FF(x) + 3/2*log(mrho/m)) - alpha/pi*c1) - 1
# ---------------------------------------------------------------------------
_alpha_pi = _alpha / np.pi
# alpha/pi * 3/2 * log(m/mrho): negative because m < mrho.
# Appears in: -alpha/pi * (-3/2*log(m/mrho) + c1) = alpha/pi*3/2*log(m/mrho) - alpha/pi*c1
_ap_3half_log_pi_rho = _alpha_pi * 1.5 * np.log(_mpi / _mrho)  # ≈ -0.00597
_ap_3half_log_K_rho = _alpha_pi * 1.5 * np.log(_mK / _mrho)  # ≈ -0.00325
_ap_FF_me_pi = _alpha_pi * _FF_me_pi
_ap_FF_mmu_pi = _alpha_pi * _FF_mmu_pi
_ap_FF_me_K = _alpha_pi * _FF_me_K
_ap_FF_mmu_K = _alpha_pi * _FF_mmu_K
_delta_c1K = 0.2 * np.log(_mK**2 / _mpi**2)  # c1K = c1p + _delta_c1K

# Kinematic phase-space factors (constants)
_kin_pi_e = _mpi * _me**2 * (1.0 - _me**2 / _mpi**2) ** 2
_kin_pi_mu = _mpi * _mmu**2 * (1.0 - _mmu**2 / _mpi**2) ** 2
_kin_K_e = _mK * _me**2 * (1.0 - _me**2 / _mK**2) ** 2
_kin_K_mu = _mK * _mmu**2 * (1.0 - _mmu**2 / _mK**2) ** 2
_pref = 1.0 / (8.0 * np.pi)  # common factor from GF^2 fP^2 Vxx^2 / (8pi) * ...

# Experimental data: [Rpi, RK, Rmu, GKmu]
_EXP = jnp.array([0.00012344, 0.00002488, 1.3367, 3.37926e-17], dtype=jnp.float64)
_SIG = jnp.array([3e-7, 9e-8, 0.00261625, 6.58212e-20], dtype=jnp.float64)

# ---------------------------------------------------------------------------
# Parameter bookkeeping
# ---------------------------------------------------------------------------
_MESON_PARAM_NAMES = {"c1p", "fkfpi", "fPK", "Sew", "Vud", "Vus"}

_DEFAULTS = {
    "c1p": -2.4,
    "fkfpi": 1.1932,
    "fPK": 0.1557,
    "Sew": 1.0232,
    "Vud": 0.9737,
    "Vus": 0.2243,
}

# The four WET LEFT WCs targeted by run_and_match (JMS basis, real parts).
# Index convention:  VnueduLL_{nu-gen}{l-gen}{u-gen}{d-gen}  where d-type: d=1, s=2.
_WET_NAMES = ("VnueduLL_1111", "VnueduLL_2211", "VnueduLL_1112", "VnueduLL_2212")


# ---------------------------------------------------------------------------
# JIT-compiled chi2 functions
# ---------------------------------------------------------------------------


@jax.jit
def _chi2_sm(c1p, fkfpi, fPK, Sew, Vud, Vus):
    """SM chi2 for leptonic meson decays (pure JAX, no SMEFT)."""
    fPpi = fPK / fkfpi
    c1K = c1p + _delta_c1K

    d_pie = Sew * (1.0 + _ap_FF_me_pi + _ap_3half_log_pi_rho - _alpha_pi * c1p) - 1.0
    d_pimu = Sew * (1.0 + _ap_FF_mmu_pi + _ap_3half_log_pi_rho - _alpha_pi * c1p) - 1.0
    d_Ke = Sew * (1.0 + _ap_FF_me_K + _ap_3half_log_K_rho - _alpha_pi * c1K) - 1.0
    d_Kmu = Sew * (1.0 + _ap_FF_mmu_K + _ap_3half_log_K_rho - _alpha_pi * c1K) - 1.0

    GF2 = _GF**2
    Gpie = GF2 * Vud**2 * fPpi**2 * _pref * _kin_pi_e * (1.0 + d_pie)
    Gpimu = GF2 * Vud**2 * fPpi**2 * _pref * _kin_pi_mu * (1.0 + d_pimu)
    GKe = GF2 * Vus**2 * fPK**2 * _pref * _kin_K_e * (1.0 + d_Ke)
    GKmu = GF2 * Vus**2 * fPK**2 * _pref * _kin_K_mu * (1.0 + d_Kmu)

    pred = jnp.array([Gpie / Gpimu, GKe / GKmu, GKmu / Gpimu, GKmu], dtype=jnp.float64)
    return jnp.sum((pred - _EXP) ** 2 / _SIG**2)


@jax.jit
def _chi2_smeft(c1p, fkfpi, fPK, Sew, Vud, Vus, L1111, L2211, L1112, L2212):
    """SMEFT chi2 for leptonic meson decays (pure JAX).

    L1111, L2211, L1112, L2212 are VnueduLL LEFT coefficients at 2 GeV (JMS).
    Convention matches chi2meson2.py: GFeff = GF*(1 - L/(2*sqrt(2)*GF)).
    """
    fPpi = fPK / fkfpi
    c1K = c1p + _delta_c1K

    d_pie = Sew * (1.0 + _ap_FF_me_pi + _ap_3half_log_pi_rho - _alpha_pi * c1p) - 1.0
    d_pimu = Sew * (1.0 + _ap_FF_mmu_pi + _ap_3half_log_pi_rho - _alpha_pi * c1p) - 1.0
    d_Ke = Sew * (1.0 + _ap_FF_me_K + _ap_3half_log_K_rho - _alpha_pi * c1K) - 1.0
    d_Kmu = Sew * (1.0 + _ap_FF_mmu_K + _ap_3half_log_K_rho - _alpha_pi * c1K) - 1.0

    sq2GF = 2.0 * jnp.sqrt(2.0) * _GF
    GFe_pi = _GF * (1.0 - L1111 / sq2GF)
    GFmu_pi = _GF * (1.0 - L2211 / sq2GF)
    GFe_K = _GF * (1.0 - L1112 / sq2GF)
    GFmu_K = _GF * (1.0 - L2212 / sq2GF)

    Gpie = GFe_pi**2 * Vud**2 * fPpi**2 * _pref * _kin_pi_e * (1.0 + d_pie)
    Gpimu = GFmu_pi**2 * Vud**2 * fPpi**2 * _pref * _kin_pi_mu * (1.0 + d_pimu)
    GKe = GFe_K**2 * Vus**2 * fPK**2 * _pref * _kin_K_e * (1.0 + d_Ke)
    GKmu = GFmu_K**2 * Vus**2 * fPK**2 * _pref * _kin_K_mu * (1.0 + d_Kmu)

    pred = jnp.array([Gpie / Gpimu, GKe / GKmu, GKmu / Gpimu, GKmu], dtype=jnp.float64)
    return jnp.sum((pred - _EXP) ** 2 / _SIG**2)


# ---------------------------------------------------------------------------
# External likelihood class
# ---------------------------------------------------------------------------


class MesonDecayChi2TwoLeptons:
    """SMEFiT external chi2 for leptonic meson decays — rgevolve Jacobian.

    Four LEFT WC Jacobians (dL_X/dc_i) are computed analytically at initialisation
    via rgevolve.tools.functions.run_and_match.  No Wilson or rgevolve calls are made
    during sampling; compute_chi2 is a pure JAX linear function of the free parameters.

    Nuisance parameters c1p, fkfpi, fPK, Sew, Vud, Vus are extracted from
    coefficient_values if declared free in the runcard; otherwise their defaults from
    chi2meson2.Gpars and PDG are used.
    """

    def __init__(self, coefficients, rge_dict=None, starting_scale=None):
        free_names = list(coefficients.free_names)

        self._nu_idx = {
            n: free_names.index(n) for n in free_names if n in _MESON_PARAM_NAMES
        }

        self._free_smeft_names = [n for n in free_names if n not in _MESON_PARAM_NAMES]
        self._smeft_idx = jnp.array(
            [free_names.index(n) for n in self._free_smeft_names], dtype=jnp.int32
        )

        if starting_scale is not None:
            self._scale = float(starting_scale)
        elif rge_dict is not None:
            self._scale = float(rge_dict.get("init_scale", 10000.0))
        else:
            self._scale = 10000.0

        self.num_data = len(_EXP)

        # dL[k] is the Jacobian vector for the k-th LEFT WC (_WET_NAMES[k])
        # shape: (4, len(free_smeft_names)) or None if no SMEFT operators are free
        if not self._free_smeft_names:
            self._dL = None
            return

        smeft_accuracy = (
            rge_dict.get("smeft_accuracy", "integrate") if rge_dict else "integrate"
        )
        adm_QCD = rge_dict.get("adm_QCD", False) if rge_dict else False
        yukawa = rge_dict.get("yukawa", "top") if rge_dict else "top"

        all_smeft_names = [n for n in coefficients.names if n not in _MESON_PARAM_NAMES]
        translation = RGE(
            all_smeft_names, self._scale, smeft_accuracy, adm_QCD, yukawa
        ).RGEbasis

        # Accumulate the effective SMEFT-param → Warsaw-WC mapping, expanding
        # constrained coefficients through their vars/expr fields.
        self._eff_translation = {n: {} for n in self._free_smeft_names}

        for coeff in coefficients.coefficients:
            if coeff.name in _MESON_PARAM_NAMES:
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

        self._dL = self._compute_jacobians()

    def _compute_jacobians(self):
        """Compute dL_X/dc_i for all four LEFT WCs via a single run_and_match call.

        Returns a JAX array of shape (4, n_free_smeft).
        Row k corresponds to _WET_NAMES[k].
        """
        warsaw_names_needed: set[str] = set()
        for wc_dict in self._eff_translation.values():
            warsaw_names_needed.update(wc_dict.keys())

        if not warsaw_names_needed:
            return jnp.zeros((4, len(self._free_smeft_names)))

        smeft_warsaw_wcs = set(wc[0] for wc in get_wc_basis("SMEFT", "Warsaw"))
        unknown = warsaw_names_needed - smeft_warsaw_wcs
        if unknown:
            warnings.warn(
                f"MesonDecayChi2: Warsaw WC names not in rgevolve's SMEFT basis "
                f"(will be ignored): {sorted(unknown)}",
                stacklevel=2,
            )
        warsaw_list = sorted(warsaw_names_needed & smeft_warsaw_wcs)

        if not warsaw_list:
            return jnp.zeros((4, len(self._free_smeft_names)))

        wcs_in = tuple((name, "R") for name in warsaw_list)
        wcs_out = tuple((name, "R") for name in _WET_NAMES)

        # M has shape (4, N_warsaw): M[k, j] = d(WET_k)/d(warsaw_j)
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

        n = len(self._free_smeft_names)
        dL = np.zeros((4, n))
        for i, name in enumerate(self._free_smeft_names):
            for j, wc in enumerate(warsaw_list):
                factor = self._eff_translation[name].get(wc, 0.0)
                if factor != 0.0:
                    dL[:, i] += M[:, j] * factor

        return jnp.array(dL)

    def compute_chi2(self, coefficient_values):
        coefficient_values = jnp.asarray(coefficient_values)

        def _get(name, default):
            idx = self._nu_idx.get(name)
            return coefficient_values[idx] if idx is not None else jnp.float64(default)

        c1p = _get("c1p", _DEFAULTS["c1p"])
        fkfpi = _get("fkfpi", _DEFAULTS["fkfpi"])
        fPK = _get("fPK", _DEFAULTS["fPK"])
        Sew = _get("Sew", _DEFAULTS["Sew"])
        Vud = _get("Vud", _DEFAULTS["Vud"])
        Vus = _get("Vus", _DEFAULTS["Vus"])

        if self._dL is None:
            return _chi2_sm(c1p, fkfpi, fPK, Sew, Vud, Vus)

        smeft_vals = coefficient_values[self._smeft_idx]
        # L[k] = dot(dL[k], smeft_vals) for k in {1111, 2211, 1112, 2212}
        L = jnp.dot(self._dL, smeft_vals)  # shape (4,)
        return _chi2_smeft(c1p, fkfpi, fPK, Sew, Vud, Vus, L[0], L[1], L[2], L[3])
