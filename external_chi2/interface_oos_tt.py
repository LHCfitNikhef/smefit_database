import numpy as np


def make_oos_tt_only_likelihood(coefficients):
    # base of WCs in the oos
    oo_tt_wc_basis = ['OpQM', 'Opt', 'OtW', 'OtZ']

    project = np.zeros((len(oo_tt_wc_basis), coefficients.size))
    for i, op in enumerate(oo_tt_wc_basis):
        if op in coefficients.name:
            project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1

    LIKELIHOODS = {
        'FCC_ee_tt_365': {'file': '/data/theorie/jthoeve/smefit_release/add_optimal_chi2/invcov_FCC_ee_tt_365GeV.dat'}
    }
    # import inverse covariance as 6x6 np.array
    invcov = np.loadtxt(LIKELIHOODS['FCC_ee_tt_365']['file'])

    proj_transp = np.transpose(project)
    chi2 = np.linalg.multi_dot([coefficients.value, proj_transp, invcov, project, coefficients.value])
    n_dat = len(oo_tt_wc_basis)
    return chi2, n_dat

