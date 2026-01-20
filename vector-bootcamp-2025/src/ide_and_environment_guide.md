# IDE and Environment Setup

### Installing VS Code Locally and Cloning the Repository

For this bootcamp, we highly recommend using VS Code as your local IDE because it makes working on the cluster GPUs
significantly easier. You can download VS Code here: https://code.visualstudio.com/

Once you have the application installed, you can clone and open a local version of the fl4health repository by
following the same set of [instructions](./repo_setup_guide.md) that you followed to download it to Vector’s cluster
but on your local machine.

See: [Repo Setup Guide](./repo_setup_guide.md)

### Setting up your Python Environment

There are comprehensive instructions for setting up your IDE and environment in the [CONTRIBUTING.MD](../CONTRIBUTING.MD). Reading and following these steps is optional, but it can be helpful if you run into issues.

You will need python 3.10 installed and available on your local machine to correctly create the python virtual
environment locally in order to use the library. If you don’t already have it, there are multiple ways to obtain a
copy and use it to create a python environment with the specific version. A few examples are:
1) Using `miniconda` following the installation instructions
([link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)) and the environment create
instructions here
2) Homebrew via this [link](https://formulae.brew.sh/formula/python@3.10).
3) Using `pyenv` following the readme here: [link](https://github.com/pyenv/pyenv/blob/master/README.md). Note that
`pyenv` can be somewhat involved to use.

Thereafter, you run the commands (or variations if your system python is not 3.10 or you're using an environment
manager like `conda`).
```bash
cd path/to/fl4health
pip install uv
uv sync --extra dev --extra test --extra codestyle
source .venv/bin/activate
```

```admonish
The environment creation step may be different depending on how 3.10 is installed on your system or whether you're
using, for example, the conda steps to create the environment.

For example, if python 3.10 is not designated as your local systems python, you may need to ensure the correct
python version is available before running `uv sync`. You can specify the python version with:
```bash
uv python install 3.10
```

If you're using `conda` then you can specify a python version to use as
```bash
conda create -n env_name python=3.10
```
where `env_name` is what you would like to call your environment. Thereafter, you would activate your environment
using
```bash
conda activate env_name
```
and proceed with the remainder of the instructions unaltered.

**Note that the above code must be run from the top level of the FL4Health directory.**

**Any time you want to run code in the library, this environment must be active.**

With uv, the environment is automatically created in `.venv/` and the command to activate it is:
```bash
source .venv/bin/activate
```

Many of the examples in the library can be run locally in a reasonable amount of time on a cpu. However, there are a
few that are much faster on a GPU. Moreover, larger models and datasets of interest may require a GPU to perform
efficient training.

### Python Environment Setup on the Cluster
For working with the library on Vector’s cluster, there are two options:
1) We have a pre-built environment that users can simply activate to start running the examples in the library and
working with our code.
2) The second option is to build your own version of the environment that you can modify to add libraries that you
would like to work with above and beyond our installations.

#### Activating and Working with Our Pre-built Environment

First log onto the cluster with
```bash
ssh username@v.vectorinstitute.ai
```
going through the steps of two-factor authentication.

The shared environment is housed in the public folder:
`/ssd003/projects/aieng/public/fl4health_bootcamp/`

All that is necessary to start working with the library is to run
```bash
source /ssd003/projects/aieng/public/fl4health_bootcamp/bin/activate
```

This should prefix your terminal code with `(fl4health_bootcamp)`

#### Creating Your Own Environment on the Cluster

If you’re going this route, you’ll need to follow the steps below to create and set up a python environment of your
own.

First log onto the cluster with
```bash
ssh username@v.vectorinstitute.ai
```
going through the steps of two-factor authentication.

The process is nearly the same as on your local machine. However, prior to installing dependencies, you will need to
activate python 3.10 on the cluster.
```bash
module load python/3.10.12
cd path/to/fl4health
pip install uv
uv sync --extra dev --extra test --extra codestyle
source .venv/bin/activate
```

### Accessing a Cluster GPU through your Local VS Code

You can also connect your local VS Code directly to a VS Code instance on a GPU or CPU on Vector’s cluster.

#### Installing VS Code Server on the Cluster

First log into the cluster with
```bash
ssh username@v.vectorinstitute.ai
```
going through the steps of two-factor authentication.

The commands below downloads and saves VSCode in your home folder on the cluster. **You need only do this once:**
```bash
cd ~/

curl -Lk 'https://update.code.visualstudio.com/1.98.2/cli-alpine-x64/stable' --output vscode_cli.tar.gz

tar -xf vscode_cli.tar.gz
rm vscode_cli.tar.gz
```

#### Setting up a Tunnel and Connecting Your Local VS Code

After logging into the cluster, run the following.
```bash
srun --gres=gpu:1 --qos=m --time=4:00:00 -c 8 --mem 16G -p t4v2 --pty bash
```

This will reserve a t4v2 GPU and provide you a terminal to run commands on that node. Note that `-p t4v2` requests
a t4v2 GPU. You can also access larger `a40` and `rtx6000` GPUs this way, but you may face longer wait times for
reservations. The `-c 8` requests 8 supporting CPUs and `--mem 16G` requests 16 GB of **CPU** memory (not GPU memory).
There may be a brief waiting period if the cluster is busy and many people are using the GPU resources.

Next verify the beginning of the command prompt to make sure that you are running this command from a GPU node
(e.g., `user@gpu001`) and not the login node (`user@v[1,2,..]`).

After that, you can spin up a tunnel to the GPU node using the following command:

```bash
~/code tunnel
```

You will be prompted to authenticate via Github. On the first run, you might also need to review Microsoft's terms of
services.

Thereafter, you will be prompted to name your tunnel. You can name it whatever you like or leave it blank and it will
default to the name of the first GPU you have connected to.

After that, you can access the tunnel through your browser (not the best but it works). If you've logged into Github
on your VSCode desktop app, you can also connect from there by installing the extension:

`ms-vscode.remote-server`

Then, in your local VS Code press Shift-Command-P (Shift-Control-P), and locate

`Remote-Tunnels: Connect to Tunnel.`

After selecting this option and waiting for VS Code to find the GPU you have started the tunnel on (under whatever
name you gave it, or the default of the first GPU you connected to), you should be able to select it. Now your VS
Code is logged into the GPU and should be able to see the file system there.

Note that you will need to keep the SSH connection running in your terminal while using the tunnel. After you are done
with the work, stop your session by pressing Control-C to release the GPU.

```admonish
GPU reservations are time limited. The command `--qos=m --time=4:00:00` guarantees that you get the GPU for 4 hours
uninterrupted. Thereafter, you may be preempted (kicked off), by other users hoping to use the resources.
```

If you want to request more time, you can increase `--time=X:00:00` to request a longer time reservation. As the
reservation time increases, so does the potential wait time to obtain the requested resources.


### Running an Example (Locally or On the Cluster)

For your convenience, we have a basic utility script that takes care of launching server and client code in
background processes, so you don’t need to worry about opening multiple terminal windows to run each client and server
process separately. It is located at

`examples/utils/run_fl_local.sh`

Of course, you may still launch processes separately and manually if you would like to.

By default, it is set up to run our basic example with 2 clients and a server. However, you may modify this script to
run other examples of your choosing. If you run (**remembering to activate your environment**)
```bash
bash examples/utils/run_fl_local.sh
```
This should kick off the federated learning processes and train a model for 2 clients using FedAvg and place the logs
in the folders specified in the script.

### Cluster Datasets

For convenience, we have stored some useful datasets on the cluster. These include datasets that your team identified
as potentially useful for the target use-cases you will be working on during the bootcamp.

These datasets are stored at `/projects/federated_learning/`.

**NOTE**: This first `/` is important. Without it the folder will not be visible to you. You can see its contents with
the command
```bash
ls /projects/federated_learning/
```

In the `/projects/federated_learning/public` folder, you will find all datasets used in the examples for the library
including MNIST, CIFAR, and others. The remainder of the folders should loosely correspond to your team names and are
populated with datasets relevant to your PoCs. You and your teammates should have access to these folders, but other
teams will not. If you cannot access your folder, please let your facilitator know and we will get it sorted out.
