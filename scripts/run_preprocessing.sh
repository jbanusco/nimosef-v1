#!/bin/bash
#SBATCH --job-name=nimosef_res
#SBATCH --output=/cluster/home/ja1659/logs/nimosef_res_%A_%a.out
#SBATCH --error=/cluster/home/ja1659/logs/nimosef_res_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --qos=16cpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --partition=rad2
#SBATCH --account=rad

# Singularity folder
singularity_path='/data/bdip2/jbanusco/SingularityImages'
singularity_img=${singularity_path}/nimosef_0.0.sif

# Dataset path
dataset_path="/data/bdip2/jbanusco/Test_NIMOSEF_Dataset"

# Code path
code_path="/cluster/home/ja1659/Code/nimosef-v1"

# Logs path
logs_folder=${dataset_path}/derivatives/nimosef_flip_logs
mkdir -p ${logs_folder}

# Docker mapping
docker_data='/usr/data'
docker_code='/usr/src'
docker_log='/usr/logs'

# Path to config file
config_file="${code_path}/nimosef/config/config_dataset_split.json"

# Loop through res_z_factor values 1 and 2
for res_z in 1 2; do
    # Extract parameters from JSON using Python and replace res_factor_z
    config_params=$(python3 -c "
import json
with open('${config_file}', 'r') as f:
    config = json.load(f)
print(' '.join(f'--{key} {value}' for key, value in config.items() if value is not None))
")
    echo "Using config parameters: ${config_params}"

    # Run inference in the dataset
    singularity exec --nv \
    --bind ${dataset_path}:${docker_data} \
    --bind ${code_path}:${docker_code} \
    --bind ${logs_folder}:${docker_log} \
    ${singularity_img} /bin/bash -c "cd ${docker_code} && python3 -m nimosef.data.preprocess_data ${config_params}"
done
