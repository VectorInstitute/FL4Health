# Introduction

This documentation serves as a guide to the distributed discrete Gaussian (DDGauss) mechanism implemented in this branch of the FL4Health repository. You will find in the following pages docs related to the implementation. 

We briefly remark here the organization of the branch.

1. The 12 experiments {distributed, central, local} $\times$ {heart, isic, ixi, tcga_brac} can be found in `research/` and can be run from the 
```sh
secure_aggregation_run_script.sh <experiment type>
```
after suitable modfication of the hyperparameter name and values.

2. In the `secure_aggregation_archive/` you will find the pseudocode, graphing, plots, custom networks we considered (i.e. not FLamby Baseline), custom FLamby pipelines to try some custom nets.

3. The core implementation are found in `fl4health/`

## Ten challenges 
1. Modular clipping causing large model parameters (solved)
2. Inefficient Welsh Hadamard transform (solved)
3. Inefficient noise sampling (solved
4. Nonconvergent rounding (solved)
5. Opacus issues (solved)
6. Implement efficient instance gradient clipping without Opacus (unsolved)
7. Batch norm, group norm non-convergence or slow convergence issues (we have some emperical observations for various FLamby tasks).
8. The effect of learning rate on privacy (unexplored).
9. Privacy accounting for mini-client approach (unsolved).
10. Large epsilons preventing privacy amplifcation (unsolved).

## Beta features
1. Mini-client approach has been implemented, but the storage is not efficient due to storing lots of models attributed to each mini-client.
2. SecAgg dropout case is not fully supported (dropout is ~90% implemented, not yet tested).

## Notes
The experiments are labeled in the sense that if you `squeue --me` you get
```
    JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
12577693       a40   DDP-id your_usrname  R       0:12      1 gpu010
```
under NAME you will see the code `DDP-id` where `-id` is short for isic distributed experiment. The tag `DDP` refers to this project which investigates distributed differential privacy. 