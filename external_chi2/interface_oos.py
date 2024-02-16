import numpy as np
import matplotlib.pyplot as plt
import scipy
import pathlib

from smefit.runner import Runner
from smefit.chi2 import Scanner
import smefit.log as log
from smefit.log import print_banner, setup_console
from smefit.optimize.ultranest import USOptimizer



def make_oos_likelihood(name:str, smeft_ops:list[str]):
    #base of WCs in the oos
    oo_wc_basis=['OpD','OpWB','OWWW','Opl1','Ope','O3pl1']
    #Check if any of the operators in the oos are being fitted.
    check_fit=any(b in smeft_ops for b in oo_wc_basis)
    if not check_fit:
        return None
    #Use this to read the inverse covariance exported from Mathematica
    LIKELIHOODS = {
        'FCC_ee_ww_lepto_161': {'file':'invcov_FCCee_ww_leptonic_161.dat'},
        'FCC_ee_ww_lepto_240': {'file':'invcov_FCCee_ww_leptonic_240.dat'},
        'FCC_ee_ww_lepto_365': {'file':'invcov_FCCee_ww_leptonic_365.dat'},
        'FCC_ee_ww_semilep_161': {'file':'invcov_FCCee_ww_semilep_161.dat'},
        'FCC_ee_ww_semilep_240': {'file':'invcov_FCCee_ww_semilep_240.dat'},
        'FCC_ee_ww_semilep_365': {'file':'invcov_FCCee_ww_semilep_365.dat'},
        'CEPC_ee_ww_lepto_161': {'file':'invcov_CEPC_ww_leptonic_161.dat'},
        'CEPC_ee_ww_lepto_240': {'file':'invcov_CEPC_ww_leptonic_240.dat'},
        'CEPC_ee_ww_lepto_365': {'file':'invcov_CEPC_ww_leptonic_365.dat'},
        'CEPC_ee_ww_semilep_161': {'file':'invcov_CEPC_ww_semilep_161.dat'},
        'CEPC_ee_ww_semilep_240': {'file':'invcov_CEPC_ww_semilep_240.dat'},
        'CEPC_ee_ww_semilep_365': {'file':'invcov_CEPC_ww_semilep_365.dat'}
    }

    if name not in LIKELIHOODS.keys():
        return None

    #import inverse covariance as 6x6 np.array
    invcov = np.loadtxt(LIKELIHOODS[name]['file'])
    #Projector from one basis to the other.
    project=np.zeros((len(oo_wc_basis),len(smeft_ops)))
    for i, op in enumerate(oo_wc_basis):
        if op in smeft_ops:
            project[i,smeft_ops.index(op)]=1
    proj_transp=np.transpose(project)

    return lambda wc: np.linalg.multi_dot([w,proj_transp,invcov,project,w])


def make_oos_tt_likelihood(name:str, smeft_ops:list[str]):
    #base of WCs in the oos
    oo_tt_wc_basis=['OpQM','Opt','OtW','OtZ']
    #Check if any of the operators in the oos are being fitted.
    check_fit=any(b in smeft_ops for b in oo_tt_wc_basis)
    if not check_fit:
        return None
    #Use this to read the inverse covariance exported from Mathematica
    LIKELIHOODS = {
        'FCC_ee_tt_365': {'file':'invcov_FCC_ee_tt_365GeV.dat'},
        'CEPC_ee_tt_365': {'file':'invcov_CEPC_ee_tt_365GeV.dat'}
    }

    if name not in LIKELIHOODS.keys():
        return None

    #import inverse covariance as 6x6 np.array
    invcov = np.loadtxt(LIKELIHOODS[name]['file'])
    #Projector from one basis to the other.
    project=np.zeros((len(oo_tt_wc_basis),len(smeft_ops)))
    for i, op in enumerate(oo_tt_wc_basis):
        if op in smeft_ops:
            project[i,smeft_ops.index(op)]=1
    proj_transp=np.transpose(project)

    return lambda wc: np.linalg.multi_dot([w,proj_transp,invcov,project,w])


def run_smefit(smefit_runcard, log_path):
    
    '''
    Parameters
    ----------
    smefit_runcard: pathlib.Path
        Path to smefit runcard
    log_path: pathlib.Path
        Path to log
    external_chi2: function
        External likelihood to add the SMEFiT likelihood
    '''
    #set up log
    setup_console(log_path)
    print_banner()

    #link SMEFiT runcard
    runner = Runner.from_file(smefit_runcard.absolute())
    log.console.log("Running: Nested Sampling fit")

    #check which operators are being fitted
    fitops= runner.run_card['coefficients'].keys()
    smeftops=[]
    for i, op in enumerate(fitops):
        smeftops.append(op) 
    #Get the optimal observables likelihood. Add further lines for other channels and energies
    ######################
    #FCC-ee likelihoods
    ######################
    #FCC-ee WW likelihoods
    smeft_chi2_fcc_lep_161 = make_oos_likelihood('FCC_ee_ww_lepto_161', smeftops)
    smeft_chi2_fcc_lep_240 = make_oos_likelihood('FCC_ee_ww_lepto_240', smeftops)
    smeft_chi2_fcc_lep_365 = make_oos_likelihood('FCC_ee_ww_lepto_365', smeftops)
    smeft_chi2_fcc_semi_161 = make_oos_likelihood('FCC_ee_ww_semilep_161', smeftops)
    smeft_chi2_fcc_semi_240 = make_oos_likelihood('FCC_ee_ww_semilep_240', smeftops)
    smeft_chi2_fcc_semi_365 = make_oos_likelihood('FCC_ee_ww_semilep_365', smeftops)
    #FCC-ee tt likelihood
    smeft_chi2_fcc_tt = make_oos_tt_likelihood('FCC_ee_tt_365', smeftops)
    #FCC-ee total OOs likelihood
    smeft_chi2_fcc_tot = smeft_chi2_fcc_lep_161 +smeft_chi2_fcc_lep_240 + smeft_chi2_fcc_lep_365 + smeft_chi2_fcc_semi_161 +smeft_chi2_fcc_semi_240 + smeft_chi2_fcc_semi_365 + smeft_chi2_fcc_tt
    ######################
    #CEPC likelihoods
    ######################
    #CEPC WW likelihoods
    smeft_chi2_cepc_lep_161 = make_oos_likelihood('CEPC_ee_ww_lepto_161', smeftops)
    smeft_chi2_cepc_lep_240 = make_oos_likelihood('CEPC_ee_ww_lepto_240', smeftops)
    smeft_chi2_cepc_lep_365 = make_oos_likelihood('CEPC_ee_ww_lepto_365', smeftops)
    smeft_chi2_cepc_semi_161 = make_oos_likelihood('CEPC_ee_ww_semilep_161', smeftops)
    smeft_chi2_cepc_semi_240 = make_oos_likelihood('CEPC_ee_ww_semilep_240', smeftops)
    smeft_chi2_cepc_semi_365 = make_oos_likelihood('CEPC_ee_ww_semilep_365', smeftops)
    #CEPC total OOs likelihood
    smeft_chi2_cepc_tt = make_oos_tt_likelihood('CEPC_ee_tt_365', smeftops)
    smeft_chi2_cepc_tot = smeft_chi2_cepc_lep_161 + smeft_chi2_cepc_lep_240 + smeft_chi2_cepc_lep_365 + smeft_chi2_cepc_semi_161 + smeft_chi2_cepc_semi_240 + smeft_chi2_cepc_semi_365 + smeft_chi2_cepc_tt

    # set up the optimizer and link the optimal observables likelihood
    opt = USOptimizer.from_dict(runner.run_card)
    opt.external_chi2 = smeft_chi2_fcc_tot + smeft_chi2_cepc_tot

    #start fit
    opt.run_sampling()

# run the fit
run_smefit(smefit_runcard=pathlib.Path("smefit_oo_test_runcard.yaml"),log_path=pathlib.Path("smefit_oo_test.log"))