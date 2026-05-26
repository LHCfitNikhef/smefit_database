"""Gaussian prior constraints on beta-decay nuisance parameters and Vud.

Each class contributes a single Gaussian term

    chi2 = ((param - central) / sigma)^2

to the total likelihood. They are designed to be used as independent entries in
the runcard's ``external_chi2`` section so they can be turned on and off
individually and their central values / widths can be set per-run.

If the target parameter is not free in the runcard (``free: false``), the class
returns 0 and has no effect — it is safe to leave entries for fixed parameters.

Example runcard entries::

    external_chi2:
      BetaDecayChi2Linear:
        path: ./external_chi2_beta_decay_linear.py
      GaussConstraintDRV:
        path: ./gaussian_constraints.py
        central: 0.02467
        sigma: 0.00022
      GaussConstraintEta2:
        path: ./gaussian_constraints.py
        central: 0.0
        sigma: 1.0
      GaussConstraintEta3:
        path: ./gaussian_constraints.py
        central: 0.0
        sigma: 1.0
      GaussConstraintVud:
        path: ./gaussian_constraints.py
        central: 0.97373
        sigma: 0.00031

Default central values and sigmas (overridable in the runcard):
  DRV  : central = 0.02467,  sigma = 0.00022   (Hardy & Towner, arXiv:2010.13797)
  eta2 : central = 0.0,      sigma = 1.0        (loose — data-driven)
  eta3 : central = 0.0,      sigma = 1.0        (loose — data-driven)
  Vud  : central = 0.97373,  sigma = 0.00031    (PDG 2022, kaon+pion decays)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class _GaussConstraintBase:
    """Base class — subclasses just set ``_param_name`` and the defaults."""

    _param_name: str  # override in each subclass

    def __init__(self, coefficients, rge_dict=None, central=None, sigma=None, **_):
        free_names = list(coefficients.free_names)
        self._idx = free_names.index(self._param_name) if self._param_name in free_names else None
        self._central = jnp.float64(float(central if central is not None else self._default_central))
        self._sigma   = jnp.float64(float(sigma   if sigma   is not None else self._default_sigma))
        self.num_data = 1

    def compute_chi2(self, coefficient_values):
        if self._idx is None:
            return jnp.float64(0.0)
        val = jnp.asarray(coefficient_values, dtype=jnp.float64)[self._idx]
        return ((val - self._central) / self._sigma) ** 2


class GaussConstraintDRV(_GaussConstraintBase):
    """Gaussian prior on DRV (universal radiative correction Delta_R^V).

    Default: central = 0.02467, sigma = 0.00022  (Hardy & Towner 2020)
    """
    _param_name = "DRV"
    _default_central = 0.02467
    _default_sigma   = 0.00022


class GaussConstraintEta2(_GaussConstraintBase):
    """Gaussian prior on eta2 (isospin-breaking nuclear correction).

    Default: central = 0.0, sigma = 1.0  (data-driven; tighten as needed)
    """
    _param_name = "eta2"
    _default_central = 0.0
    _default_sigma   = 1.0


class GaussConstraintEta3(_GaussConstraintBase):
    """Gaussian prior on eta3 (Q-value-dependent nuclear correction).

    Default: central = 0.0, sigma = 1.0  (data-driven; tighten as needed)
    """
    _param_name = "eta3"
    _default_central = 0.0
    _default_sigma   = 1.0


class GaussConstraintVud(_GaussConstraintBase):
    """Gaussian prior on Vud from kaon/pion decays (external to beta decays).

    Default: central = 0.97373, sigma = 0.00031  (PDG 2022)
    """
    _param_name = "Vud"
    _default_central = 0.97373
    _default_sigma   = 0.00031
