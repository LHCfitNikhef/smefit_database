# -*- coding: utf-8 -*-
import pathlib

import numpy as np
import pandas as pd
import yaml

from ..compute_theory import make_predictions
from ..covmat import covmat_from_systematics
from ..loader import Loader, load_datasets
from ..log import logging

_logger = logging.getLogger(__name__)


class Projection:
    def __init__(
        self,
        commondata_path,
        theory_path,
        dataset_names,
        projections_path,
        coefficients,
        order,
        use_quad,
        rot_to_fit_basis,
        fred_tot,
        fred_sys,
    ):

        self.commondata_path = commondata_path
        self.theory_path = theory_path
        self.dataset_names = dataset_names
        self.projections_path = projections_path
        self.coefficients = coefficients
        self.order = order
        self.use_quad = use_quad
        self.rot_to_fit_basis = rot_to_fit_basis
        self.fred_tot = fred_tot
        self.fred_sys = fred_sys

        self.datasets = load_datasets(
            self.commondata_path,
            self.dataset_names,
            self.coefficients,
            self.order,
            self.use_quad,
            False,
            False,
            False,
            theory_path=self.theory_path,
        )

        if self.coefficients:
            _logger.info(
                f"Some coefficients are specified in the runcard: EFT correction will be used for the central values"
            )

    @classmethod
    def from_config(cls, projection_card):
        with open(projection_card, encoding="utf-8") as f:
            projection_config = yaml.safe_load(f)

        commondata_path = pathlib.Path(projection_config["commondata_path"]).absolute()
        theory_path = pathlib.Path(projection_config["theory_path"]).absolute()
        projections_path = pathlib.Path(
            projection_config["projections_path"]
        ).absolute()
        dataset_names = projection_config["datasets"]

        coefficients = projection_config.get("coefficients", [])
        order = projection_config.get("order", "LO")
        use_quad = projection_config.get("use_quad", False)
        rot_to_fit_basis = projection_config.get("rot_to_fit_basis", None)

        fred_tot = projection_config.get("fred_tot", 1)
        fred_sys = projection_config.get("fred_sys", 1)

        return cls(
            commondata_path,
            theory_path,
            dataset_names,
            projections_path,
            coefficients,
            order,
            use_quad,
            rot_to_fit_basis,
            fred_tot,
            fred_sys
        )

    def compute_cv_projection(self):
        """
        Computes the new central value under the EFT hypothesis (is SM when coefficients are zero)

        Returns
        -------
            cv : numpy.ndarray
                SM + EFT theory predictions
        """
        cv = self.datasets.SMTheory

        if self.coefficients:
            coefficient_values = []
            for coeff in self.datasets.OperatorsNames:
                coefficient_values.append(self.coefficients[coeff]["value"])
            cv = make_predictions(
                self.datasets, coefficient_values, self.use_quad, False
            )
        return cv

    def rescale_sys(self, sys, fred_sys):
        """
        Projects the systematic uncertainties by reducing them by fred_sys

        Parameters
        ----------
        sys: asystematics
        fred: reduction factor

        Returns
        -------

        """

        # check whether the systematics are artificial (i.e. no breakdown of separate systematic sources),
        # characterised by a square matrix and negative values
        is_square = sys.shape[0] == sys.shape[1]
        is_artificial = is_square & np.any(sys < 0)

        if is_artificial:

            # reconstruct covmat and keep only diagonal components
            cov_tot = sys @ sys.T
            sys_diag = np.sqrt(np.diagonal(cov_tot))

            # rescale systematics and make df
            sys_rescaled = np.diag(sys_diag * fred_sys)
            return pd.DataFrame(sys_rescaled, index=sys.index, columns=sys.columns)

        return sys * fred_sys

    @staticmethod
    def rescale_stat(stat, lumi_old, lumi_new):

        """
        Projects the statistical uncertainties from lumi_old to lumi_new

        Parameters
        ----------
        stat: array_like
            old statistical uncertainties
        lumi_old: float
            Old luminosity
        lumi_new: float
            New luminosity

        Returns
        -------
        Updated statistical uncertainties after projection
        """
        fred_stat = np.sqrt(lumi_old / lumi_new)
        return stat * fred_stat

    def build_projection(self, lumi_new):
        """
        Constructs runcard for projection by updating the central value and statistical uncertainties

        Parameters
        ----------
        lumi_new: float
            Adjusts the statistical uncertainties according to the specified luminosity
        """

        # compute central values under projection
        cv = self.compute_cv_projection()

        cnt = 0
        for dataset_idx, ndat in enumerate(self.datasets.NdataExp):

            dataset_name = self.datasets.ExpNames[dataset_idx]
            path_to_dataset = self.commondata_path / f"{dataset_name}.yaml"

            _logger.info(f"Building projection for : {dataset_name}")

            with open(path_to_dataset, encoding="utf-8") as f:
                data_dict = yaml.safe_load(f)

            idxs = slice(cnt, cnt + ndat)

            # load the statistical and systematic uncertainties
            stat = np.asarray(data_dict["statistical_error"])

            if not isinstance(data_dict["sys_names"], list):  # for a single systematic
                sys = pd.DataFrame(data_dict["systematics"], [data_dict["sys_names"]]).T
            else:
                sys = pd.DataFrame(data_dict["systematics"], data_dict["sys_names"]).T

            # if all stats are zero, we only have access to the total error which we rescale by 1/3 (compromise)
            no_stats = not np.any(stat)
            if no_stats:
                fred = self.fred_tot
                sys_red = self.rescale_sys(sys, fred)
                stat_red = stat
            # if separate stat and sys
            else:
                fred = self.fred_sys
                lumi_old = self.datasets.Luminosity[dataset_idx]
                stat_red = self.rescale_stat(stat, lumi_old, lumi_new)
                sys_red = self.rescale_sys(sys, fred)

            data_dict["systematics"] = sys_red.values.tolist()
            data_dict["statistical_error"] = stat_red.tolist()

            # build covmat for projections. Use rescaled stat
            newcov = covmat_from_systematics([stat_red], [sys_red])


            # add L1 noise to cv
            cv_projection = np.random.multivariate_normal(cv[idxs], newcov)

            # replace cv with updated central values
            if len(cv_projection) > 1:
                data_dict["data_central"] = cv_projection.tolist()
            else:
                data_dict["data_central"] = float(cv_projection[0])

            projection_folder = self.projections_path
            projection_folder.mkdir(exist_ok=True)
            with open(f"{projection_folder}/{dataset_name}_proj.yaml", "w") as file:
                yaml.dump(data_dict, file, sort_keys=False)

            cnt += ndat
