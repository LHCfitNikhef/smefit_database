# -*- coding: utf-8 -*-
import pathlib
import argparse

import numpy as np
import yaml

new_commondata_path = pathlib.Path(__file__).parent
new_commondata_folder = pathlib.Path(f"{new_commondata_path}/commondata")
new_commondata_folder.mkdir(exist_ok=True)


def load_data(dataset):

    data_file = old_commondata_path / f"DATA_{dataset}.dat"
    sys_file = old_commondata_path / f"systypes/SYSTYPE_{dataset}_DEFAULT.dat"

    num_sys, num_data = np.loadtxt(data_file, usecols=(1, 2), max_rows=1, dtype=int)

    central_values, stat_error = np.loadtxt(
        data_file, usecols=(5, 6), unpack=True, skiprows=1
    )

    # Load systematics from commondata file.
    # Read values of sys first
    sys_add = []
    sys_mult = []
    for i in range(0, num_sys):
        add, mult = np.loadtxt(
            data_file,
            usecols=(7 + 2 * i, 8 + 2 * i),
            unpack=True,
            skiprows=1,
        )
        sys_add.append(add)
        sys_mult.append(mult)

    sys_add = np.asarray(sys_add)
    sys_mult = np.asarray(sys_mult)

    # Read systype file
    if num_sys != 0:
        type_sys, name_sys = np.genfromtxt(
            sys_file,
            usecols=(1, 2),
            unpack=True,
            skip_header=1,
            dtype="str",
        )
    else:
        type_sys = np.array([])
        name_sys = np.array([])
    return central_values, num_data, num_sys, stat_error, sys_add, type_sys, name_sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate old experiemntal files into new format"
    )
    parser.add_argument(
        "old_commondata_path", type=pathlib.Path, help="path to old commondata folder"
    )
    args = parser.parse_args()
    old_commondata_path = args.old_commondata_path
    datasets = [
        "ATLAS_tt_8TeV_ljets_Mtt",
        "ATLAS_tt_8TeV_dilep_Mtt",
        "CMS_tt_8TeV_ljets_Ytt",
        "CMS_tt2D_8TeV_dilep_MttYtt",
        "CMS_tt_13TeV_ljets_2015_Mtt",
        "CMS_tt_13TeV_dilep_2015_Mtt",
        "CMS_tt_13TeV_ljets_2016_Mtt",
        "CMS_tt_13TeV_dilep_2016_Mtt",
        "ATLAS_tt_13TeV_ljets_2016_Mtt",
        "ATLAS_CMS_tt_AC_8TeV",
        "ATLAS_tt_AC_13TeV",
        # ttbar asymm and helicity frac
        "ATLAS_WhelF_8TeV",
        "CMS_WhelF_8TeV",
        # ttbb
        "CMS_ttbb_13TeV",
        "CMS_ttbb_13TeV_2016",
        "ATLAS_ttbb_13TeV_2016",
        # tttt
        "CMS_tttt_13TeV",
        "CMS_tttt_13TeV_run2",
        "ATLAS_tttt_13TeV_run2",
        # ttZ
        "CMS_ttZ_8TeV",
        "CMS_ttZ_13TeV",
        "CMS_ttZ_13TeV_pTZ",
        "ATLAS_ttZ_8TeV",
        "ATLAS_ttZ_13TeV",
        "ATLAS_ttZ_13TeV_2016",
        # ttW
        "CMS_ttW_8TeV",
        "CMS_ttW_13TeV",
        "ATLAS_ttW_8TeV",
        "ATLAS_ttW_13TeV",
        "ATLAS_ttW_13TeV_2016",
        # Single top
        "CMS_t_tch_8TeV_inc",
        "ATLAS_t_tch_8TeV",
        "CMS_t_tch_8TeV_diff_Yt",
        "CMS_t_sch_8TeV",
        "ATLAS_t_sch_8TeV",
        "ATLAS_t_tch_13TeV",
        "CMS_t_tch_13TeV_inc",
        "CMS_t_tch_13TeV_diff_Yt",
        "CMS_t_tch_13TeV_2016_diff_Yt",
        # tW
        "ATLAS_tW_8TeV_inc",
        "ATLAS_tW_slep_8TeV_inc",
        "CMS_tW_8TeV_inc",
        "ATLAS_tW_13TeV_inc",
        "CMS_tW_13TeV_inc",
        # tZ
        "ATLAS_tZ_13TeV_inc",
        "ATLAS_tZ_13TeV_run2_inc",
        "CMS_tZ_13TeV_inc",
        "CMS_tZ_13TeV_2016_inc",
        # HIGGS PRODUCTION
        # ATLAS & CMS Combined Run 1 Higgs Measurements
        "ATLAS_CMS_SSinc_RunI",
        "ATLAS_SSinc_RunII",
        "CMS_SSinc_RunII",
        # ATLAS & CMS Run II Higgs Differential
        "CMS_H_13TeV_2015_pTH",
        "ATLAS_H_13TeV_2015_pTH",
        # ATLAS & CMS STXS
        "ATLAS_WH_Hbb_13TeV",
        "ATLAS_ZH_Hbb_13TeV",
        "ATLAS_ggF_ZZ_13TeV",
        "CMS_ggF_aa_13TeV",
        # DIBOSON DATA
        "ATLAS_WW_13TeV_2016_memu",
        "ATLAS_WZ_13TeV_2016_mTWZ",
        "CMS_WZ_13TeV_2016_pTZ",
        # LEP
        "LEP_eeWW_182GeV",
        "LEP_eeWW_189GeV",
        "LEP_eeWW_198GeV",
        "LEP_eeWW_206GeV",
    ]

    for dataset in datasets:

        print(f"Converting dataset: {dataset}")
        (
            central_values,
            num_data,
            num_sys,
            stat_error,
            sys_add,
            type_sys,
            name_sys,
        ) = load_data(dataset)
        exp_name = {"dataset_name": dataset}
        num_data = {"num_data": num_data.tolist()}
        num_sys = {"num_sys": num_sys.tolist()}
        data_central_yaml = {"data_central": central_values.tolist()}
        stat = {"statistical_error": stat_error.tolist()}
        sys = {"systematics": sys_add.tolist()}
        sys_names = {"sys_names": name_sys.tolist()}
        sys_type = {"sys_type": type_sys.tolist()}

        new_commondata_folder = pathlib.Path(f"{new_commondata_path}/commondata")
        new_commondata_folder.mkdir(exist_ok=True)
        with open(f"{new_commondata_folder}/{dataset}.yaml", "w") as file:
            yaml.dump(exp_name, file, sort_keys=False)
            yaml.dump(num_data, file, sort_keys=False)
            yaml.dump(num_sys, file, sort_keys=False)
            yaml.dump(data_central_yaml, file, sort_keys=False)
            yaml.dump(stat, file, sort_keys=False)
            yaml.dump(sys, file, sort_keys=False)
            yaml.dump(sys_names, file, sort_keys=False)
            yaml.dump(sys_type, file, sort_keys=False)
