"""Gaussian prior constraints on meson-decay nuisance parameters.

Each class contributes a single Gaussian term

    chi2 = ((param - central) / sigma)^2

to the total likelihood. They are designed to be used as independent entries in
the runcard's ``external_chi2`` section so they can be turned on and off
individually and their central values / widths can be set per-run.

If the target parameter is not free in the runcard (``free: false``), the class
returns 0 and has no effect — it is safe to leave entries for fixed parameters.

Example runcard entries::

    external_chi2:
      MesonDecayChi2:
        path: ./chi2meson2.py
      GaussConstraintC1p:
        path: ./gaussian_nuisance_meson_decay_2leptons.py
      GaussConstraintFkfpi:
        path: ./gaussian_nuisance_meson_decay_2leptons.py
      GaussConstraintFPK:
        path: ./gaussian_nuisance_meson_decay_2leptons.py
      GaussConstraintSew:
        path: ./gaussian_nuisance_meson_decay_2leptons.py

Default central values and sigmas (overridable in the runcard):
  c1p   : central = -2.4,    sigma = 0.5       (EM chiral counterterm)
  fkfpi : central = 1.1932,  sigma = 0.0021    (FK/Fpi ratio)
  fPK   : central = 0.1557,  sigma = 0.0003    (kaon decay constant FK, GeV)
  Sew   : central = 1.0232,  sigma = 0.0003    (short-distance EM correction)
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
        self._idx = (
            free_names.index(self._param_name)
            if self._param_name in free_names
            else None
        )
        self._central = jnp.float64(
            float(central if central is not None else self._default_central)
        )
        self._sigma = jnp.float64(
            float(sigma if sigma is not None else self._default_sigma)
        )
        self.num_data = 1

    def compute_chi2(self, coefficient_values):
        if self._idx is None:
            return jnp.float64(0.0)
        val = jnp.asarray(coefficient_values, dtype=jnp.float64)[self._idx]
        return ((val - self._central) / self._sigma) ** 2


class GaussConstraintC1p(_GaussConstraintBase):
    """Gaussian prior on c1p (EM chiral counterterm in pi/K leptonic decays).

    Default: central = -2.4, sigma = 0.5
    """

    _param_name = "c1p"
    _default_central = -2.4
    _default_sigma = 0.5


class GaussConstraintFkfpi(_GaussConstraintBase):
    """Gaussian prior on fkfpi (ratio of kaon to pion decay constants FK/Fpi).

    Default: central = 1.1932, sigma = 0.0021
    """

    _param_name = "fkfpi"
    _default_central = 1.1932
    _default_sigma = 0.0021


class GaussConstraintFPK(_GaussConstraintBase):
    """Gaussian prior on fPK (kaon decay constant FK, in GeV).

    Default: central = 0.1557, sigma = 0.0003
    """

    _param_name = "fPK"
    _default_central = 0.1557
    _default_sigma = 0.0003


class GaussConstraintSew(_GaussConstraintBase):
    """Gaussian prior on Sew (short-distance electromagnetic correction factor).

    Default: central = 1.0232, sigma = 0.0003
    """

    _param_name = "Sew"
    _default_central = 1.0232
    _default_sigma = 0.0003
