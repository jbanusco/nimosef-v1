#!/bin/bash
#SBATCH --job-name=nimosef_res
#SBATCH --output=/cluster/home/ja1659/logs/nimosef_res_%A_%a.out
#SBATCH --error=/cluster/home/ja1659/logs/nimosef_res_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
##SBATCH --gres=gpu:1
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --partition=rad
#SBATCH --account=rad
#SBATCH --exclude=gpunode01

# Singularity folder
singularity_path='/data/bdip2/jbanusco/SingularityImages'
singularity_img=${singularity_path}/nimosef_0.0.sif

# Dataset path
dataset_path="/data/bdip2/jbanusco/UKB_Cardiac_BIDS"

# Code path
code_path='/cluster/home/ja1659/Code/nimosef'

# Logs path
logs_folder=${dataset_path}/derivatives/nimosef_flip_logs
mkdir -p ${logs_folder}

# Docker mapping
docker_data='/usr/data'
docker_code='/usr/src'
docker_log='/usr/logs'

# Path to config file
config_file="${code_path}/configs/generate_results_config_baseline_v1.json"
#config_file="${code_path}/configs/generate_results_config_motion_v1.json"

# Loop through res_z_factor values 1 and 2
for res_z in 1 2; do
    # Extract parameters from JSON using Python and replace res_factor_z
    config_params=$(python3 -c "
import json
with open('${config_file}', 'r') as f:
    config = json.load(f)
config['res_factor_z'] = ${res_z}  # Overwrite res_factor_z
if ${res_z} == 2:
    config['num_subjects'] = 10  # Set num_subjects to 10 when res_factor_z is 2
print(' '.join(f'--{key} {value}' for key, value in config.items() if value is not None))
")
    echo "Running with res_factor_z=${res_z}"
    echo "Using config parameters: ${config_params}"

    # Run inference in the dataset
    singularity exec --nv \
    --bind ${dataset_path}:${docker_data} \
    --bind ${code_path}/src:${docker_code} \
    --bind ${logs_folder}:${docker_log} \
    ${singularity_img} /bin/bash -c "cd ${docker_code} && python -m nimosef.training.generate_results ${config_params}"
done
