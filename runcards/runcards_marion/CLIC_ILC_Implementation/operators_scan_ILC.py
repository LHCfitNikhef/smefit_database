from itertools import combinations
import os, sys


print(sys.argv)
job_id = sys.argv[0]

operators = ["cpDC", "cpWB", "cWWW","cpl1", "c3pl1", "c3pl2","cpe","cll1221"]

op_pairs = list(combinations(operators, 2))

op_list = operators + op_pairs


def write_op(ops, file):
    for op in operators:
        if len(ops) == 2:
            if op in ops:
                file.write("set "+op+" 1\n")
            else:
                file.write("set "+op+" 1e-9\n")
        else:
            if op == ops:
                file.write("set "+op+" 1\n")
            else:
                file.write("set "+op+" 1e-9\n")


def write_settings(file):

    file.write("analysis=off\n")
    file.write("set fixed_ren_scale True\n")
    file.write("set fixed_fac_scale True\n")
    file.write("set gf 1.16637870e-05\n")
    file.write("set mt 1.720000e+02\n")
    file.write("set mz 9.118760e+01\n")
    file.write("set mw 8.038700e+01\n")
    file.write("set mh 1.250000e+02\n")
    file.write("set nevents 100000\n")
    file.write("set ebeam1 250\n") #####CHANGE BEAM ENERGY
    file.write("set ebeam2 250\n") #####CHANGE BEAM ENERGY
    

with open("scan_ops_eeWW_ILC.mg5".format(job_id), 'w') as script:

    script.write("#/home/MG5_aMC_v3_4_1/bin/mg5_aMC scan_ops_eeWW_ILC.mg5\n")
     
    script.write("launch /home/marion/eeWW_SM --name=init_run3\n")
    write_settings(script)
    
    script.write("launch /home/marion/eeWW_NP2 --name=init_run3\n")
    write_settings(script)
    
    script.write("launch /home/marion/eeWW_NP4 --name=init_run3\n")
    write_settings(script)
  
    script.write("launch /home/marion/eeWW_SM --name=SM_ILC_500GeV_pos80_neg30\n")
    script.write("analysis=off\n")
    script.write("set polbeam2 0.8\n")
    script.write("set polbeam1 -0.3\n")     

    script.write("launch /home/marion/eeWW_SM --name=SM_ILC_500GeV_neg80_pos30\n")
    script.write("analysis=off\n")
    script.write("set polbeam2 -0.8\n")
    script.write("set polbeam1 0.3\n") 
    
    for op in op_list:
        if len(op) == 2:
            if os.path.isdir("./Events/%s" % "_".join(op)):
                continue
            script.write("launch /home/marion/eeWW_NP4 --name=%s_ILC_500GeV_pos80_neg30_sq\n" % "_".join(op))
            script.write("analysis=off\n")
            script.write("set polbeam2 0.8\n")
            script.write("set polbeam1 -0.3\n")
            write_op(op, script)
        else:
            if os.path.isdir("./Events/%s" % op):
                continue
            script.write("launch /home/marion/eeWW_NP2 --name=%s_ILC_500GeV_pos80_neg30_inter\n" % op)
            script.write("analysis=off\n")
            script.write("set polbeam2 0.8\n")
            script.write("set polbeam1 -0.3\n")
            write_op(op, script)
            
            script.write("launch /home/marion/eeWW_NP4 --name=%s_ILC_500GeV_pos80_neg30_sq\n" % op)
            script.write("analysis=off\n")
            script.write("set polbeam2 0.8\n")
            script.write("set polbeam1 -0.3\n")
            write_op(op, script)
            
    for op in op_list:
        if len(op) == 2:
            if os.path.isdir("./Events/%s" % "_".join(op)):
                continue
            script.write("launch /home/marion/eeWW_NP4 --name=%s_ILC_500GeV_neg80_pos30_sq\n" % "_".join(op))
            script.write("analysis=off\n")
            script.write("set polbeam2 -0.8\n")
            script.write("set polbeam1 0.3\n")
            write_op(op, script)
        else:
            if os.path.isdir("./Events/%s" % op):
                continue
            script.write("launch /home/marion/eeWW_NP2 --name=%s_ILC_500GeV_neg80_pos30_inter\n" % op)
            script.write("analysis=off\n")
            script.write("set polbeam2 -0.8\n")
            script.write("set polbeam1 0.3\n")
            write_op(op, script)
            
            script.write("launch /home/marion/eeWW_NP4 --name=%s_ILC_500GeV_neg80_pos30_sq\n" % op)
            script.write("analysis=off\n")
            script.write("set polbeam2 -0.8\n")
            script.write("set polbeam1 0.3\n")
            write_op(op, script)
