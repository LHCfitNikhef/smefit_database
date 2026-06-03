from external_chi2.optimal_observables.interface_oos_lep3 import (
    oo_ww_wc_basis,
)

EXPECTED_WW_WC_BASIS = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]


def test_ww_oo_wc_basis_values():
    assert oo_ww_wc_basis == EXPECTED_WW_WC_BASIS
