# Introduction

This documentation serves as a guide to the distributed discrete Gaussian (DDGauss) mechanism implemented in this branch of the FL4Health repository. You will find in the following pages the various documentation to the implementation. 

We briefly remark here the organization of the branch.

1. The 12 experiments can be found in `research/` which can be run from the 
```sh
secure_aggregation_run_script.sh
```
2. In the `secure_aggregation_archive/` you will find the pseudocode, graphing, plots, custom networks we considered (i.e. not FLamby Baseline), custom FLamby pipelines to try some custom nets.

## Ten challenges 
1. Modular clipping causing large numbers (solved)
2. Inefficient Welsh Hadamard transform (solved)
3. Inefficient noise sampling (accelerated)
4. Nonconvergent rounding (solved)
5. Opacus issues (solved)
6. Implement efficient instance gradient clipping without Opacus (unsolved)
7. Batch norm, group norm non-convergence or slow convergence issues (we have some emperical observations for various FLamby tasks).
8. The effect of learning rate on privacy (unexplored).
9. Privacy accounting for mini-client approach (unsolved).
10. Large epsilons preventing privacy amplifcation (unsolved).

## Beta features
1. Mini-client approach has been implemented, but the storage is not efficient due to storing lots of models attributed to each mini-client.
2. SecAgg dropout case is not fully supported (it is ~90% implemented, not yet tested).

