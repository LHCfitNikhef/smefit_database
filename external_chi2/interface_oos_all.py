import numpy as np


def make_oos_ww_chi2(coefficients):

    # #base of WCs in the oos

    oo_wc_basis = ['OpD', 'OpWB', 'OWWW', 'Opl1', 'Ope', 'O3pl1']

    project = np.zeros((len(oo_wc_basis), coefficients.size))
    for i, op in enumerate(oo_wc_basis):
        if op in coefficients.name:
            project[i, np.argwhere(coefficients.name == op)[0, 0]] = 1


    datasets = {
        'FCC_ee_ww_lepto_161': '/data/theorie/jthoeve/smefit_release/add_optimal_chi2/invcov_FCCee_ww_leptonic_161.dat',
        'FCC_ee_ww_lepto_240': '/data/theorie/jthoeve/smefit_release/add_optimal_chi2/invcov_FCCee_ww_leptonic_240.dat',
        'FCC_ee_ww_lepto_365': '/data/theorie/jthoeve/smefit_release/add_optimal_chi2/invcov_FCCee_ww_leptonic_365.dat',
        'FCC_ee_ww_semilep_161': '/data/theorie/jthoeve/smefit_release/add_optimal_chi2/invcov_FCCee_ww_semilep_161.dat',
        'FCC_ee_ww_semilep_240': '/data/theorie/jthoeve/smefit_release/add_optimal_chi2/invcov_FCCee_ww_semilep_240.dat',
        'FCC_ee_ww_semilep_365': '/data/theorie/jthoeve/smefit_release/add_optimal_chi2/invcov_FCCee_ww_semilep_365.dat'
    }

    chi2 = 0
    n_dat = len(oo_wc_basis)

    for dataset, invcov_path in datasets.items():
        # #import inverse covariance as 6x6 np.array
        invcov = np.loadtxt(invcov_path)
        chi2 += np.linalg.multi_dot([coefficients.value, project.T, invcov, project, coefficients.value])

    return chi2, n_dat


def make_oos_tt_chi2(coefficients):
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



