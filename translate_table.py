# -*- coding: utf-8 -*-
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

new_table_path = pathlib.Path(__file__).parent


def load_data(dataset):
    """Load theory data"""
    out_dict = {}

    # read best sm
    with open(f"{old_table_path}/theory/{dataset}.txt", "r", encoding="utf-8") as f:
        best_sm = f.readlines()
    f.close()
    out_dict["best_sm"] = np.array([float(v) for v in best_sm[0].split()])

    # read theory cov
    with open(f"{old_table_path}/theory/{dataset}_cov.txt", "r", encoding="utf-8") as f:
        th_cov = []
        for line in f.readlines():
            th_cov.append([float(v) for v in line.split()])
    f.close()
    out_dict["theory_cov"] = np.array(th_cov)

    # read operator_res tables
    for order in ["LO", "NLO"]:
        op_res = {}
        with open(
            f"{old_table_path}/operator_res/{order}/{dataset}.txt",
            "r",
            encoding="utf-8",
        ) as f:
            for line in f.readlines()[1:]:
                key, *val = line.split()
                op_res[key] = np.array([float(v) for v in val])
        f.close()
        out_dict[order] = change_label_names(pd.DataFrame.from_dict(op_res).T)

    return out_dict


def change_label_names(df):
    for idx in df.index:
        if "^" in idx:
            op = idx.split("^")[0]
            df = df.rename(index={idx: f"{op}*{op}"})
    return df


def dump_to_json(data_dict, dataset):
    new_table_folder = pathlib.Path(f"{new_table_path}/theory/")
    new_table_folder.mkdir(exist_ok=True)
    with open(f"{new_table_folder}/{dataset}.json", "w", encoding="utf-8") as f:
        data_dict["best_sm"] = data_dict["best_sm"].tolist()
        data_dict["theory_cov"] = data_dict["theory_cov"].tolist()
        data_dict["LO"] = (data_dict["LO"].T).to_dict("list")
        data_dict["NLO"] = (data_dict["NLO"].T).to_dict("list")
        json.dump(data_dict, f, indent=" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate old theory files into new format"
    )
    parser.add_argument(
        "old_table_path", type=pathlib.Path, help="path to old tables folder"
    )
    args = parser.parse_args()
    old_table_path = args.old_table_path
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
    for d in datasets:
        print(f"Converting dataset: {d}")
        data_dict = load_data(d)
        dump_to_json(data_dict, d)
