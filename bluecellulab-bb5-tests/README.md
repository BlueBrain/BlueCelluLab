# bluecellulab-bb5-tests

## Description

This folder contains tests for bluecellulab on the large circuits located on bb5. Please note that these tests are not directly usable as they are due to dependencies.

## How to run locally

Simply clone the repository, install tox and type `tox` in the root directory of the repository.

## Continous Integration

This module will continue to be part of the continuous integration (CI) pipeline on GitLab until the end of the year. The scheduled CI tasks will run the tests for this module periodically until 2024.12.31. After this date, the tests will no longer be included in the CI pipeline.

## Mechanisms

The mechanisms for these tests have been merged into the [tests/mechanisms/](./../tests/mechanisms/) folder. Here is the list of mechanisms used for these tests. The original list is provided below, and files marked with an asterisk (*) were added to the tests folder as they were not already present:
- `Ca.mod`
- `CaDynamics_DC0.mod`
- `CaDynamics_E2.mod`
- `Ca_HVA.mod`
- `Ca_HVA2.mod`
- `Ca_LVAst.mod`
- `DetAMPANMDA.mod`
- `DetGABAAB.mod`
- `GluSynapse.mod`
- *`IN_Ih_Halnes2011.mod`
- *`IN_iT.mod`
- `Ih.mod`
- `Im.mod`
- `K_Pst.mod`
- `K_Tst.mod`
- `KdShu2007.mod`
- `NaTa_t.mod`
- `NaTg.mod`
- `NaTs2_t.mod`
- `Nap_Et2.mod`
- `ProbAMPANMDA_EMS.mod`
- `ProbGABAAB_EMS.mod`
- *`RC_IT_Des92.mod`
- `SK_E2.mod`
- `SKv3_1.mod`
- `StochKv.mod`
- `StochKv3.mod`
- *`TC_HH.mod`
- *`TC_ITGHK_Des98.mod`
- *`TC_Ih_Bud97.mod`
- *`TC_Ih_CaMod.mod`
- *`TC_Kleak.mod`
- *`TC_Naleak.mod`
- *`TC_Nap_Et2.mod`
- *`TC_cadecay.mod`
- *`TC_iA.mod`
- *`TC_iL.mod`
- *`TC_kir_Con15.mod`
- `TTXDynamicsSwitch.mod`
- `VecStim.mod`
- `gap.mod`
- *`ican.mod`
- `netstim_inhpoisson.mod`
