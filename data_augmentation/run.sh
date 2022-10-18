#!/bin/bash
export CONFIG_DIR="/accounts/projects/jsteinhardt/$USER/src/rusp/data_augmentation"
export LM_DIR="/accounts/projects/jsteinhardt/$USER/src/language_models"
export CONDA_ENV_PREFIX="/scratch/users/$USER/conda/common"

source /usr/local/linux/anaconda3.8/etc/profile.d/conda.sh

conda activate $CONDA_ENV_PREFIX

for EXPERIMENT in "$@"
do
	echo $EXPERIMENT
	python $LM_DIR/language_models.py \
		--config=$CONFIG_DIR/config.py:$EXPERIMENT
done