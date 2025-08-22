from external_chi2.optimal_observables.interface_oos_cepc import (
    oo_ww_wc_basis,
    oo_tt_wc_basis,
)

EXPECTED_WW_WC_BASIS = ["OpD", "OpWB", "OWWW", "Opl1", "Ope", "O3pl1"]
EXPECTED_TT_WC_BASIS = ["OpQM", "Opt", "OtW", "OtZ"]


def test_ww_oo_wc_basis_values():
    assert oo_ww_wc_basis == EXPECTED_WW_WC_BASIS


def test_tt_oo_wc_basis_values():
    assert oo_tt_wc_basis == EXPECTED_TT_WC_BASIS
