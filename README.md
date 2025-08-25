# SMEFiT Database

This repository collects the **experimental datasets** and **theoretical calculations** used in the SMEFiT global analysis framework.  

<!-- BEGIN: SMEFT-OPERATORS -->
## SMEFiT Wilson coefficients

Definitions are given in terms of the [Warsaw basis (WCxf)](https://wcxf.github.io/assets/pdf/SMEFT.Warsaw.pdf).

This section is auto-generated from `operators_implemented.yaml`. Do not edit it manually.

Each entry defines the Wilson coefficient `cX`, corresponding to the SMEFiT operator `OX`, as used in the JSON theory tables.

**Operator index conventions:**
- `i` denotes the first two right-handed up-quark and left-handed quark doublet generations (i = 1, 2), assumed identical.
- `j` denotes the three right-handed down-quark generations (j = 1, 2, 3), assumed identical.

These conventions reflect the flavour-symmetry assumptions adopted in SMEFiT.

### Bosonic

```text
cp = phi
cpG = phiG
cpB = phiB
cpW = phiW
cpWB = phiWB
cpBox = phiBox
cpD = phiD
cWWW = W
```

### Dipoles

```text
ctG = 1/gs * uG_33
ctW = uW_33
ctZ = cw * uW_33 - sw * uB_33
```

### Quark currents

```text
cpQM = phiq1_33 - phiq3_33
c3pQ3 = phiq3_33
cpt = phiu_33
cpqMi = phiq1_ii - phiq3_ii
c3pq = phiq3_ii
cpui = phiu_ii
cpdi = phid_jj
```

### Yukawa

```text
ctp = uphi_33
ccp = uphi_22
cbp = dphi_33
ctap = ephi_33
```

### 4Heavy four-quarks

```text
cQQ1 = 2 * qq1_3333 - 2/3 * qq3_3333
cQQ8 = 8 * qq3_3333
cQt1 = qu1_3333
cQt8 = qu8_3333
ctt1 = uu_3333
```

### 4Heavy (right-handed bottom, only included in tttt and ttbb datasets)

```text
cQb1 = qd1_3333
cQb8 = qd8_3333
ctb1 = ud1_3333
ctb8 = ud8_3333
cQtQb1 = quqd1_3333
cQtQb8 = quqd8_3333
cbb = dd_3333
```

### 2L2H four-quarks

```text
c81qq = qq1_i33i + 3 * qq3_i33i
c11qq = qq1_ii33 + 1/6 * qq1_i33i + 1/2 * qq3_i33i
c83qq = qq1_i33i - qq3_i33i
c13qq = qq3_ii33 + 1/6 * qq1_i33i - 1/6 * qq3_i33i
c8qt = qu8_ii33
c1qt = qu1_ii33
c8ut = 2 * uu_i33i
c1ut = uu_ii33 + 1/3 * uu_i33i
c8qu = qu8_33ii
c1qu = qu1_33ii
c8dt = ud8_33jj
c1dt = ud1_33jj
c8qd = qd8_33jj
c1qd = qd1_33jj
```

### Lepton currents

```text
cpl1 = phil1_11
cpl2 = phil1_22
cpl3 = phil1_33
c3pl1 = phil3_11
c3pl2 = phil3_22
c3pl3 = phil3_33
cpe = phie_11
cpmu = phie_22
cpta = phie_33
```

### Four-lepton

```text
cll1111 = ll_1111
cll2222 = ll_2222
cll3333 = ll_3333
cll1122 = 1/2 * ll_1122
cll1133 = 1/2 * ll_1133
cll2233 = 1/2 * ll_2233
cll1221 = 1/2 * ll_1221
cll1331 = 1/2 * ll_1331
cll2332 = 1/2 * ll_2332
cle1111 = le_1111
cle2222 = le_2222
cle3333 = le_3333
cle3311 = le_3311
cle3322 = le_3322
cle1133 = le_1133
cle2211 = le_2211
cle1221 = le_1221
cle1122 = le_1122
cle2233 = le_2233
cee1111 = ee_1111
cee2222 = ee_2222
cee3333 = ee_3333
cee1133 = 1/4 * ee_1133
cee1122 = 1/4 * ee_1122
cee2233 = 1/4 * ee_2233
```

### 2L2Q operators

```text
cQl1M = lq1_1133 - lq3_1133
cQl13 = lq3_1133
cQl3M = lq1_3333 - lq3_3333
cQl33 = lq3_3333
cQe = qe_3311
cQta = qe_3333
ctl1 = lu_1133
ctl2 = lu_2233
ctl3 = lu_3333
cte = eu_1133
ctta = eu_3333
```

### 2L2q operators

```text
cql1M = lq1_11ii - lq3_11ii
cql13 = lq3_11ii
cqe = qe_ii11
cl1u = lu_11ii
cl1b = ld_1133
ceb = ed_1133
```


<!-- END: SMEFT-OPERATORS -->