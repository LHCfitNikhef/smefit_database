"""Gaussian prior constraints on beta-decay nuisance parameters and Vud.

Each class contributes a single Gaussian term

    chi2 = ((param - central) / sigma)^2

to the total likelihood.

Example runcard entries::

    external_chi2:
      SA_beta_decays:
        path: smefit_database/external_chi2/low_energy/superallowed_beta_decay.py
      GaussConstraintDRV:
        path: smefit_database/external_chi2/low_energy/gaussian_constraints.py
        central: 0.02467
        sigma: 0.00022

Default central values and sigmas (overridable in the runcard):
  DRV  : central = 0.02467,  sigma = 0.00022   (Hardy & Towner, arXiv:2010.13797)
  eta1 : central = 0.0,      sigma = 1.0        (loose — data-driven)
  eta2 : central = 0.0,      sigma = 1.0        (loose — data-driven)
  eta3 : central = 0.0,      sigma = 1.0        (loose — data-driven)
  Vud  : central = 0.97373,  sigma = 0.00031    (PDG 2022, kaon+pion decays)
"""

import jax.numpy as jnp


class _GaussConstraintBase:
    """Base class — subclasses just set _param_name and the defaults."""

    _param_name = None  # override in each subclass

    def __init__(self, coefficients, rge_dict=None, central=None, sigma=None, **_):
        """
        coefficients: The Wilson coefficients to be used in the analysis.
        rge_dict: A dictionary containing the RGE information (unused).
        central: Central value of the Gaussian. Defaults to the class value.
        sigma: Width of the Gaussian. Defaults to the class value.
        """
        names = [str(n) for n in coefficients.name]
        self._idx = names.index(self._param_name) if self._param_name in names else None
        self._central = float(central if central is not None else self._default_central)
        self._sigma = float(sigma if sigma is not None else self._default_sigma)
        self.n_dat = 1

    def compute_chi2(self, coefficient_values):
        # coefficient_values is the full coefficient vector.
        if self._idx is None:
            return 0.0
        val = coefficient_values[self._idx]
        return ((val - self._central) / self._sigma) ** 2


class GaussConstraintDRV(_GaussConstraintBase):
    """Gaussian prior on DRV (universal radiative correction Delta_R^V).

    Default: central = 0.02467, sigma = 0.00022  (Hardy & Towner 2020)
    """

    _param_name = "DRV"
    _default_central = 0.02467
    _default_sigma = 0.00022


class GaussConstraintEta1(_GaussConstraintBase):
    """Gaussian prior on eta1 (isospin-breaking nuclear correction).

    Default: central = 0.0, sigma = 1.0  (data-driven; tighten as needed)
    """

    _param_name = "eta1"
    _default_central = 0.0
    _default_sigma = 1.0


class GaussConstraintEta2(_GaussConstraintBase):
    """Gaussian prior on eta2 (isospin-breaking nuclear correction).

    Default: central = 0.0, sigma = 1.0  (data-driven; tighten as needed)
    """

    _param_name = "eta2"
    _default_central = 0.0
    _default_sigma = 1.0


class GaussConstraintEta3(_GaussConstraintBase):
    """Gaussian prior on eta3 (Q-value-dependent nuclear correction).

    Default: central = 0.0, sigma = 1.0  (data-driven; tighten as needed)
    """

    _param_name = "eta3"
    _default_central = 0.0
    _default_sigma = 1.0


class GaussConstraintVud(_GaussConstraintBase):
    """Gaussian prior on Vud from kaon/pion decays (external to beta decays).

    Default: central = 0.97373, sigma = 0.00031  (PDG 2022)
    """

    _param_name = "Vud"
    _default_central = 0.97373
    _default_sigma = 0.00031
