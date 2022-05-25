# -*- coding: utf-8 -*-
"""Test compute_theory and chi2 module"""
import numpy as np

from smefit import chi2, compute_theory, loader


def test_flatten():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    flatten_matrix = np.array([1, 2, 3, 5, 6, 9])
    np.testing.assert_allclose(compute_theory.flatten(matrix), flatten_matrix)


class TestPredictions:
    exp_data = np.array(
        [
            1.3,
            2.6,
        ]
    )
    sm_theory = np.array(
        [
            1,
            1,
        ]
    )
    operators_names = np.array(["Op1", "Op2"])
    lin_corr_values = np.array(
        [
            [0.1, 0.2],
            [0.1, 0.2],
        ]
    )  # Op1, Op2
    quad_corr_values = np.array(
        [
            [0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5],
        ]
    )  # Op1^2, Op1*Op2, Op2^2
    covmat = np.diag(np.ones(2))
    dataset = loader.DataTuple(
        exp_data,
        sm_theory,
        operators_names,
        lin_corr_values,
        quad_corr_values,
        np.array(["exp1"]),
        exp_data.size,
        np.linalg.inv(covmat),
    )
    wilson_coeff = np.array([0.5, 0.6])

    def test_make_predictions(self):

        lin_corr = [
            # data0
            self.sm_theory[0]
            + self.wilson_coeff[0] * self.lin_corr_values[0, 0]
            + self.wilson_coeff[1] * self.lin_corr_values[0, 1],
            # data1
            self.sm_theory[1]
            + self.wilson_coeff[0] * self.lin_corr_values[1, 0]
            + self.wilson_coeff[1] * self.lin_corr_values[1, 1],
        ]
        np.testing.assert_allclose(
            compute_theory.make_predictions(self.dataset, self.wilson_coeff, False),
            lin_corr,
        )
        quad_corr = np.array(
            [
                # data0
                +self.wilson_coeff[0] ** 2 * self.quad_corr_values[0, 0]
                + self.wilson_coeff[0]
                * self.wilson_coeff[1]
                * self.quad_corr_values[0, 1]
                + self.wilson_coeff[1] ** 2 * self.quad_corr_values[0, 2],
                # data1
                +self.wilson_coeff[0] ** 2 * self.quad_corr_values[1, 0]
                + self.wilson_coeff[0]
                * self.wilson_coeff[1]
                * self.quad_corr_values[1, 1]
                + self.wilson_coeff[1] ** 2 * self.quad_corr_values[1, 2],
            ]
        )
        quad_corr += lin_corr
        np.testing.assert_allclose(
            compute_theory.make_predictions(self.dataset, self.wilson_coeff, True),
            quad_corr,
        )

    def test_compute_chi2(self):
        diff = self.exp_data - compute_theory.make_predictions(
            self.dataset, self.wilson_coeff, True
        )
        np.testing.assert_allclose(
            chi2.compute_chi2(self.dataset, self.wilson_coeff, True),
            diff @ np.linalg.inv(self.covmat) @ diff,
        )
