#!/bin/bash
src_path=/home/jaume/Desktop/Code/nimosef-v1
data_path=/media/jaume/DATA/Data/Test_NIMOSEF_Dataset

cd ${src_path}

python -m nimosef.data.preprocess_data \
  --root ${data_path} \
  --patients -1 \
  --save-dir implicit \
  --manifest-dir manifests_nimosef \
  --use-roi

data_path=/media/jaume/DATA/Data/Urblauna_SFTP/UKB_Cardiac_BIDS
python -m nimosef.data.preprocess_data \
  --root ${data_path} \
  --patients -1 \
  --save-dir implicit \
  --manifest-dir manifests_nimosef \
  --use-roi

# Check the manifest file
cat ${data_path}/derivatives/manifests_nimosef/dataset_manifest.json

# Run training
python -m tests.run_train

# Tensorboard
tensorboard --logdir ${data_path}/derivatives/nimosef_v1 --port 6006

# Run inference
python -m tests.run_inference

splits_filename=${data_path}/derivatives/manifests_nimosef/dataset_manifest.json
model_filename=${data_path}/derivatives/nimosef_v1/experiment_20250907_183826/model.pth
model_filename=${data_path}/derivatives/nimosef_v1/experiment_20250908_223712/model.pth
model_filename=${data_path}/derivatives/nimosef_v1/experiment_20250909_104844/model.pth

# Generate results
results_folder=${data_path}/derivatives/nimosef_v1_results
python -m nimosef.training.generate_results \
    --data_folder ${data_path} \
    --split_file ${splits_filename} \
    --mode train \
    --model_to_rec ${model_filename} \
    --save_rec_folders ${results_folder} \
    --res_factor_z 1.0 \
    --overwrite_imgs True \
    --save_motion_corrected True

# L2 distance and mean shape code
python -m nimosef.analysis.save_shape_code \
    --data_folder ${data_path} \
    --split_file ${splits_filename} \
    --mode train \
    --model_to_rec ${model_filename} \
    --save_rec_folders ${results_folder} \
    --res_factor_z 1.0 \
    --overwrite_imgs True    

# Save mean image
python -m nimosef.analysis.save_mean_img \
    --data_folder ${data_path} \
    --split_file ${splits_filename} \
    --mode train \
    --model_to_rec ${model_filename} \
    --save_rec_folders ${results_folder} \
    --res_factor_z 1.0 \
    --overwrite_imgs True   

# Higher resolution
results_folder=${data_path}/derivatives/nimosef_v1_results_hr
python -m nimosef.training.generate_results \
    --data_folder ${data_path} \
    --split_file ${splits_filename} \
    --mode train \
    --model_to_rec ${model_filename} \
    --save_rec_folders ${results_folder} \
    --res_factor_z 2.0 \
    --overwrite_imgs True \
    --save_motion_corrected True

# pytest -q tests/test_preprocess_pipeline.py
# pytest -q -m realdata
# pytest -q tests/test_losses_base.py
# pytest -q tests/test_losses_utils.py 
# pytest -q tests/test_losses_composite.py 
# pytest -q tests/test_metrics.py 
# pytest -q tests/test_nimosef_model.py
# pytest -q tests/test_utils_core.py
# pytest -q tests/test_utils_visualization.py 
# pytest -q tests/test_utils_viz3d.py
# pytest -q tests/test_logging.py
# pytest -q tests/test_schedulers.py
