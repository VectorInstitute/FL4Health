# Monai Scripts

This directory contains work to integrate the Monai AutoSeg3d pipeline

### AutoSeg3d.py

This file runs the monai autoseg3d pipeline on an nnunet structured dataset.

Autoseg3d is designed to work with the very common [MSD](http://medicaldecathlon.com/) dataset format. The nnunet derives it's dataset format from the MSD guidelines but alters it slightly. One of the main changes is that different modalities/channels are stored as different files. Although nnunet provides a script to convert an MSD dataset into an nnunet dataset, we'd rather not have multiple local copies of the same raw dataset. It's easier to get monai's autoseg3d to work with nnunet datasets than to get nnunet to work with MSD datasets, therefore we choose the nnunet dataset structure as our standard that will work with everything.

Use the ```--help``` flag for a list of arguments that can be passed to the autoseg3d.py script. To run the default autoseg3d pipeline on a nnunet structured dataset run the following command

```bash
python autoseg3d.py --data-dir path/to/nnunet_raw/dataset --output-dir store/pipeline/outputs/here
```

The above command will run the full pipeline which can take a long time and be computationally expensive. We encourage running a test experiment of the full pipeline first to check for errors or issues. This can be done by creating a yaml file to redefine num_epochs, num_epochs_per_validation and num_warmup_epochs. See [train_params.yaml](train_params.yaml) for an example. You can further speed up your test experiment by using a dummy dataset with just a few samples, just make sure your batch size is not too large with num_images_per_batch. An example invocation is below

```bash
python autoseg3d.py --data-dir path/to/data --output-dir /store/output/here --train-params path/to/train_params.yaml
```
