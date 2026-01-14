### Installing the Flamby dependencies

__NOTE__: The workflow below is normally the smoothest way to construct the FLamby + FL4Health environment required to run the FLamby experiments. However, with a recent upgrade to MonAI, some of the functionality that FLamby depends on are broken. Until this is fixed, the workflow below will not work.

First clone the FLamby repository
``` bash
git clone https://github.com/owkin/FLamby.git
```
Create a python environment with your preferred env manager. We'll use conda below
``` bash
conda create -n flamby_fl4health python=3.10
conda activate flamby_fl4health
```
Install the FL4Health requirements
``` bash
cd <fl4health_repository>
pip install uv
uv sync --extra dev --extra test --extra codestyle
cd <FLamby_repository>
pip install albumentations==1.4.20
pip install -e ".[cam16, heart, isic2019, ixi, lidc, tcga]"
```
__NOTE__: The forced install of albumentations 1.4.20 avoids an issue where versions > 1.4.20 rely on simsimd which does not compile on the Vector cluster
__NOTE__: We avoid installing Fed-KITS2019, as it requires a fairly old version on nnUnet, which we no longer support in our library.

In addition, you'll have to edit the code for `FedIXITiny` in `flamby/datasets/fed_ixi/datasets.py` replacing the following

<table>
<tr>
<th>Old</th>
<th>New</th>
</tr>
<tr>
<td>

``` python
from monai.transforms import (
    AddChannel,
    ...
```
</td>
<td>

``` python
from monai.transforms import (
    EnsureChannelFirst,
    ...
```
</td>
</tr>
</tr>
<tr>
<td>

```python
default_transform = Compose(
    [ToTensor(), AddChannel(), Resize(self.common_shape)]
```
</td>
<td>

```python
default_transform = Compose(
    [ToTensor(), EnsureChannelFirst(channel_dim="no_channel"), Resize(self.common_shape)]
)
```
</td>
</tr>
</table>

This is because AddChannel was removed in Version 1.3 of MonAI

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

The following command will reserve an A40 on the cluster in order to debug experiments and launch hyperparameter sweeps. Note that it is not necessary to use an interactive A40 for the latter.

```bash
srun --gres=gpu:1 -c 8 --mem 16G -p a40 --pty bash
```
