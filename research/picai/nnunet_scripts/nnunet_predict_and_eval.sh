#!/bin/bash

# SBATCH --gres=gpu:1
# SBATCH --qos=normal
# SBATCH --mem=32GB
# SBATCH -c 12
# SBATCH --time=12:00:00
# SBATCH --job-name=nnunet_predict_and_eval
# SBATCH --output=nnunet_predict_and_eval%j.out
# SBATCH --error=nnunet_predict_and_eval%j.out

# Process Inputs
CONFIG_PATH=${1:-"${HOME}/FL4Health/research/picai/nnunet_scripts/pred_config.yaml"}
INPUT_FOLDER=${2:-"${HOME}/haider_lab/711/imagesTr/"}
LABEL_FOLDER=${3:-"${HOME}/haider_lab/711/labelsTr/"}
OUTPUT_FOLDER=${4:-"/checkpoint/jewtay/13807166"}
VENV_PATH=${5:-"/h/jewtay/.cache/pypoetry/virtualenvs/fl4health-MVmwdY85-py3.10"}

# Log input information
echo "Config Path: ${CONFIG_PATH}"
echo "INPUT FOLDER: ${INPUT_FOLDER}"
echo "OUTPUT FOLDER: ${OUTPUT_FOLDER}"
echo "LABEL FOLDER: ${LABEL_FOLDER}"
echo "Python Env Path: ${VENV_PATH}"

# Setup before running training
source ${VENV_PATH}bin/activate

python eval.py --probs-path "${OUTPUT_FOLDER}/predicted_probability_maps" --gt-path $LABEL_FOLDER --output-path "${OUTPUT_FOLDER}/output/"
