# Bootcamp Repository Setup

```admonish
These instructions assume you have already followed the basic Vector cluster account setup, including initial access,
changing your password, and setting up multifactor authentication. These instructions have been sent to you by the
Vector Ops Team. Multifactor authentication is now required upon all connections to the Vector cluster.
```

### Overview

In this section, you will create ssh keys on the Vector cluster in order to connect to the
[Fl4Health](https://github.com/VectorInstitute/FL4Health) GitHub repository. You will need to add these ssh keys to
your GitHub profile in order to clone the repository and access code. A similar process may be followed on your
local machine to establish keys to clone the repository locally.

### Creating Your SSH Keys

First, login to Vaughan (Vector cluster) over ssh using your login credentials (replace username with your own Vector
username). If you are using Windows, use Windows PowerShell to run local commands, including the following one. An
alternative for Windows is to use [git-bash](https://gitforwindows.org/). Otherwise, use Terminal.

```bash
ssh username@v.vectorinstitute.ai
```

Once logged into the Vaughan cluster, create ssh keys
(**replace your_email@example.com with your GitHub account email address**). For additional reference, see
information
[here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

When prompted to choose a file in which to save the key, just press Enter for the default. Additionally, when asked
to enter a passphrase, press Enter to proceed without setting a passphrase. It is alright not to set one.

Using the command below show your public key in the terminal and copy it to the clipboard (replace username with your
own Vector cluster username)

```bash
cat /h/$USER/.ssh/id_ed25519.pub
```

Add this ssh key to your GitHub profile by following the steps on this page:
[Add New SSH Key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

### Cloning the Repository

Return to your terminal session and clone the fl4health repository into your home directory.

```bash
cd ~/
git clone git@github.com:/VectorInstitute/fl4health.git
```

There should be a new folder in your home directory called FL4Health.

Once you have successfully cloned the fl4health repository, please proceed to setting up your VS Code and Python
Environment. These steps are outlined [ide_and_environment_guide.md](./ide_and_environment_guide.md)
