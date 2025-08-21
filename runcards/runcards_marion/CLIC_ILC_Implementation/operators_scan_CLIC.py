from itertools import combinations
import os, sys


print(sys.argv)
job_id = sys.argv[0]

operators = ["cpDC", "cpWB", "cWWW", "cpl1", "c3pl1", "c3pl2", "cpe", "cll1221"]

op_pairs = list(combinations(operators, 2))

op_list = operators + op_pairs


def write_op(ops, file):
    for op in operators:
        if len(ops) == 2:
            if op in ops:
                file.write("set " + op + " 1\n")
            else:
                file.write("set " + op + " 1e-9\n")
        else:
            if op == ops:
                file.write("set " + op + " 1\n")
            else:
                file.write("set " + op + " 1e-9\n")


def write_settings(file):
    file.write("analysis=off\n")
    file.write("set fixed_ren_scale True\n")
    file.write("set fixed_fac_scale True\n")
    file.write("set gf 1.16637870e-05\n")
    file.write("set mt 1.720000e+02\n")
    file.write("set mz 9.118760e+01\n")
    file.write("set mw 8.038700e+01\n")
    file.write("set mh 1.250000e+02\n")
    file.write("set nevents 500000\n")
    file.write("set ebeam1 1500\n")  #####CHANGE BEAM ENERGY
    file.write("set ebeam2 1500\n")  #####CHANGE BEAM ENERGY


with open("scan_ops_eeWW_CLIC3000.mg5".format(job_id), "w") as script:
    script.write("#/home/MG5_aMC_v3_4_1/bin/mg5_aMC scan_ops_eeWW_CLIC3000.mg5\n")

    script.write("launch eeWW_SM_CLIC3000 --name=init_run1\n")
    write_settings(script)

    script.write("launch eeWW_NP2_CLIC3000 --name=init_run1\n")
    write_settings(script)

    script.write("launch eeWW_NP4_CLIC3000 --name=init_run1\n")
    write_settings(script)

    script.write("launch eeWW_SM_CLIC3000 --name=SM_CLICv2_1500GeV_pos80\n")
    script.write("analysis=off\n")
    script.write("set polbeam2 0.8\n")
    script.write("set polbeam1 0.0\n")

    script.write("launch eeWW_SM_CLIC3000 --name=SM_CLICv2_1500GeV_neg80\n")
    script.write("analysis=off\n")
    script.write("set polbeam2 -0.8\n")
    script.write("set polbeam1 0.0\n")

    for op in op_list:
        if len(op) == 2:
            if os.path.isdir("./Events/%s" % "_".join(op)):
                continue
            script.write(
                "launch eeWW_NP4_CLIC3000 --name=%s_CLICv2_1500GeV_pos80_sq\n"
                % "_".join(op)
            )
            script.write("analysis=off\n")
            script.write("set polbeam2 0.8\n")
            script.write("set polbeam1 0.0\n")
            write_op(op, script)
        else:
            if os.path.isdir("./Events/%s" % op):
                continue
            script.write(
                "launch eeWW_NP2_CLIC3000 --name=%s_CLICv2_1500GeV_pos80_inter\n" % op
            )
            script.write("analysis=off\n")
            script.write("set polbeam2 0.8\n")
            script.write("set polbeam1 0.0\n")
            write_op(op, script)

            script.write(
                "launch eeWW_NP4_CLIC3000 --name=%s_CLICv2_1500GeV_pos80_sq\n" % op
            )
            script.write("analysis=off\n")
            script.write("set polbeam2 0.8\n")
            script.write("set polbeam1 0.0\n")
            write_op(op, script)

    for op in op_list:
        if len(op) == 2:
            if os.path.isdir("./Events/%s" % "_".join(op)):
                continue
            script.write(
                "launch eeWW_NP4_CLIC3000 --name=%s_CLICv2_1500GeV_neg80_sq\n"
                % "_".join(op)
            )
            script.write("analysis=off\n")
            script.write("set polbeam2 -0.8\n")
            script.write("set polbeam1 0.0\n")
            write_op(op, script)
        else:
            if os.path.isdir("./Events/%s" % op):
                continue
            script.write(
                "launch eeWW_NP2_CLIC3000 --name=%s_CLICv2_1500GeV_neg80_inter\n" % op
            )
            script.write("analysis=off\n")
            script.write("set polbeam2 -0.8\n")
            script.write("set polbeam1 0.0\n")
            write_op(op, script)

            script.write(
                "launch eeWW_NP4_CLIC3000 --name=%s_CLICv2_1500GeV_neg80_sq\n" % op
            )
            script.write("analysis=off\n")
            script.write("set polbeam2 -0.8\n")
            script.write("set polbeam1 0.0\n")
            write_op(op, script)
