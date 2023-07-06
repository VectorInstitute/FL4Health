### Installing the Flamby dependencies

First clone the FLamby repository
``` bash
git clone https://github.com/owkin/FLamby.git
```
Create a python environment with your preferred env manager. We'll use conda below
``` bash
conda create -n flamby_fl4health python=3.9
conda activate flamby_fl4health
```
Install the FL4Health requirements
``` bash
cd <fl4health_repository>
pip install -r requirements.txt
cd <FLamby_repository>
pip install -e ".[all_extra]"
```

### Downloading the Fed ISIC 2019 Dataset

After cloning the repository and installing the environment, you'll need to download and preprocess the dataset so that it can be loaded by the dataloaders. First navigate to the FLamby repository folder and do the following
``` bash
cd flamby/datasets/fed_isic2019/dataset_creation_scripts
python download_isic.py --output-folder /path/to/user/folder
```
Note that you'll have to agree to a user license and the `path/to/user/folder` that you provide will be stored for the next step.
```bash
python resize_images.py
```
For more discussion of this process, see [FLamby documentation](https://owkin.github.io/FLamby/fed_isic.html#)

### Downloading the Fed Heart Disease Dataset

After cloning the repository and installing the environment, you'll need to download and preprocess the dataset so that it can be loaded by the dataloaders. First navigate to the FLamby repository folder and do the following
``` bash
cd flamby/datasets/fed_heart_disease/dataset_creation_scripts
python download.py --output-folder /path/to/user/folder
```
Note that you'll have to agree to a user license and the `path/to/user/folder` that you provide will be stored

For more discussion of this process, see [FLamby documentation](https://owkin.github.io/FLamby/fed_heart.html)

### Downloading the Fed IXI Dataset

After cloning the repository and installing the environment, you'll need to download and preprocess the dataset so that it can be loaded by the dataloaders. First navigate to the FLamby repository folder and do the following
``` bash
cd flamby/datasets/fed_ixi/dataset_creation_scripts
python download.py --output-folder /path/to/user/folder
```
Note that you'll have to agree to a user license and the `path/to/user/folder` that you provide will be stored.

For more discussion of this process, see [FLamby documentation](https://owkin.github.io/FLamby/fed_ixi.html#)

### Getting a GPU to debug/launch experiments from

The following command will reserve an A40 on the cluster in order to debug experiments and launch hyperparameter sweeps. Note that it is not necessary to use an interactive A40 for the later.

```bash
srun --gres=gpu:1 -c 8 --mem 16G -p a40 --pty bash
```
