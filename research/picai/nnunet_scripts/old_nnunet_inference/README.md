# Old nnunet inference

This folder contains an old implementation of inference and evaluation for nnunet models

It loads the dataset into memory as opposed to reading and writing files to disk.
The new implementation uses functions available from nnunet and picai eval.

Keeping the old version here as a legacy in case we ever decide to implement our
own inference/evaluation pipeline that similar to the current implementation,
doesn't require loading the whole dataset into memory, but also doesn't have to
read and write files a bunch of times. All this additional file loading and writing
adds a significant amount of overhead. For example, with the picai dataset and a
two 2d nnunet models, inference takes 0.2s/case per model, and additional 0.2s/case
is added as a result of the additional file reads and writes.
