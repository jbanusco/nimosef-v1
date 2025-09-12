import sys
import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.profiler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from nimosef.models.nimosef import MultiHeadNetwork
from nimosef.data.dataset import NiftiDataset, nifti_collate_fn
from nimosef.losses.composite import CompositeLoss


def get_options():        
    data_folder = '/media/jaume/DATA/Data/Test_NIMOSEF_Dataset'
    split_filename = '/media/jaume/DATA/Data/Test_NIMOSEF_Dataset/derivatives/manifests_nimosef/dataset_manifest.json'
    dataset_split = 'train'

    number_patients = 4
    save_folder = os.path.join(data_folder, "derivatives", "nimosef")
    save_profile_folder = os.path.join(save_folder, "profiler")
    model_filename = os.path.join(save_folder, "experiment_20250906_140021", "checkpoint_80.pth")
    
    lr_shape_code = 1e-3
    lr = 5e-4
    init_epoch = 0
    num_epochs = 2
    sample_percentage = 0.9
    batch_size = 2

    options = {
        'save_profile_folder': save_profile_folder,
        'save_folder': save_folder,
        'data_folder': data_folder,
        'split_filename': split_filename,
        'dataset_split': dataset_split,
        'number_patients': number_patients,
        'model_filename': model_filename,
        'lr_shape_code': lr_shape_code,
        'lr': lr,
        'init_epoch': init_epoch,
        'num_epochs': num_epochs,
        'sample_percentage': sample_percentage,
        'batch_size': batch_size
    }

    return options

def main():
    # =============================================================================
    # Dataset Preparation
    # =============================================================================
    options = get_options()
    save_folder = options['save_folder']
    data_folder = options['data_folder']
    split_filename = options['split_filename']
    dataset_split = options['dataset_split']
    number_patients = options['number_patients']
    model_filename = options['model_filename']
    lr_shape_code = options['lr_shape_code']
    lr = options['lr']
    init_epoch = options['init_epoch']
    num_epochs = options['num_epochs']
    sample_percentage = options['sample_percentage']
    batch_size = options['batch_size']
    save_profile_folder = options['save_profile_folder']
    print(data_folder)
    os.makedirs(save_profile_folder, exist_ok=True)

    print(f"Using split: {dataset_split}")
    dataset = NiftiDataset(split_filename, mode=dataset_split)

    # Retrieve subject list from dataset
    subjects = dataset.patients
    
    # Create a subset containing only the first subject
    subset_indices = [0]
    subset_dataset = Subset(dataset, subset_indices)
    # Copy custom attributes from the original dataset that are needed.
    if hasattr(dataset, 'patients'):
        subset_dataset.patients = [dataset.patients[i] for i in subset_indices]

    # Create DataLoader with custom collate function
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,        
        pin_memory=True,
        prefetch_factor=None,
        persistent_workers=False,
        collate_fn=nifti_collate_fn,
    )

    # =============================================================================
    # Model, Optimizer, and Loss Setup
    # =============================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model parameters
    num_labels = 4  # 3 + 1 for background
    latent_size = 128
    motion_size = 64
    hidden_size = 128
    num_res_layers = 8
    linear_head = True

    num_subjects = len(dataset)
    model = MultiHeadNetwork(num_subjects, num_labels, latent_size, motion_size, 
                             hidden_size=hidden_size, num_res_layers=num_res_layers,
                             linear_head=linear_head)

    # Load pre-trained model weights
    state_dict = torch.load(model_filename)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Set up the optimizer with separate learning rates for shape_code and other parameters
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'shape_code' not in n and p.requires_grad], 'lr': lr},
        {'params': model.shape_code.parameters(), 'lr': lr_shape_code}
    ], weight_decay=0)

    # Instantiate loss module
    # Log hyperparameters
    # hparams = {
    #     "latent_size": args.latent_size,
    #     "motion_size": args.motion_size,
    #     "hidden_size": args.hidden_size,
    #     "num_res_layers": args.num_res_layers,
    #     "linear_head": args.linear_head,
    #     "lr": args.lr,
    #     "lr_shape_code": args.lr_shape_code,
    #     "lambda_rec": args.lambda_rec,
    #     "lambda_seg": args.lambda_seg,
    #     "lambda_reg": args.lambda_reg,
    #     "lambda_dsp": args.lambda_dsp,
    #     "lambda_reg_dsp": args.lambda_reg_dsp,
    #     "lambda_jacobian": args.lambda_jacobian,
    #     "lambda_graph_conn": args.lambda_graph_conn,
    #     "lambda_smoothness": args.lambda_smoothness,
    #     "warmup_epochs": args.warmup_epochs,
    #     "max_dsp_weight": args.max_dsp_weight,
    # }
    # loss_module = CompositeLoss(is_test=False, device=device, hp_dict=hparams)
    loss_module = CompositeLoss(is_test=False, device=device)

    # Create a GradScaler for mixed precision training
    scaler = torch.amp.GradScaler(device=device)

    # =============================================================================
    # Training Loop with Profiling
    # =============================================================================
    model.train()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(save_profile_folder),
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)  
    ) as prof:

        for epoch in range(init_epoch, num_epochs):
            epoch_start_time = time.time()            
            model.train()

            for batch in tqdm(data_loader, desc=f"Epoch {epoch} batches", unit="batch"):
                optimizer.zero_grad()

                with torch.amp.autocast(device.type):
                    # === Unpack batch data ===
                    coords, t0_data, tgt_data, t, transform_info, sample_id = batch
                    num_points = coords.shape[0]
                    idx_points = torch.randperm(num_points)[:int(sample_percentage * num_points)]

                    t0_intensity, t0_seg, t0_mask = t0_data
                    tgt_intensity, tgt_seg, tgt_mask = tgt_data

                    coords = coords[idx_points].to(device)
                    t0_intensity = t0_intensity[idx_points].to(device)
                    t0_seg = t0_seg[idx_points].to(device)
                    t0_mask = t0_mask[idx_points].to(device)
                    tgt_intensity = tgt_intensity[idx_points].to(device)
                    tgt_seg = tgt_seg[idx_points].to(device)
                    tgt_mask = tgt_mask[idx_points].to(device)
                    t = t[idx_points].to(device)
                    sample_id = sample_id[idx_points].to(device)

                    # === Forward t0 ===
                    t0 = torch.zeros((coords.shape[0], 1), device=device)
                    seg_pred_t0, int_pred_t0, disp_t0, _ = model(coords, t0, sample_id)
                    
                    # === Forward target ===
                    t = t.unsqueeze(-1)
                    seg_pred_t, int_pred_t, disp_t, _ = model(coords, t, sample_id)

                    # === Loss ===
                    # Get transform info for rescaling
                    centers = transform_info['centers'][0].to(device)
                    affine = transform_info['affines'][0].to(device)
                    indices = transform_info['indices'][0].to(device)

                    # BUT, IS THIS PER SUBJECT?
                    original_resolution = torch.linalg.norm(affine[:3, :3], axis=0) # Voxel resolution

                    h = model.shape_code(torch.unique(sample_id))
                    preds_t0 = {'seg_pred': seg_pred_t0, 'intensity_pred': int_pred_t0, 'displacement': disp_t0, 'h': h, 'batch_size': batch_size}
                    preds_t = {'seg_pred': seg_pred_t, 'intensity_pred': int_pred_t, 'displacement': disp_t, 'h': h, 'batch_size': batch_size}
                    targets_t0 = {'segmentation': t0_seg, 'intensity': t0_intensity, 'mask': t0_mask, 'batch_size': batch_size}
                    targets_t = {'segmentation': tgt_seg, 'intensity': tgt_intensity, 'mask': tgt_mask, 'batch_size': batch_size}

                    total_loss, loss_components = loss_module(sample_id.to(device), coords.to(device), original_resolution,
                                                            preds_t0, preds_t, targets_t0, targets_t, epoch=epoch)

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # advance profiler
                prof.step()

        # === Print summary after each epoch ===
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        # Optionally export the trace for further inspection
        # prof.export_chrome_trace(os.path.join(save_profile_folder, f"trace_epoch{epoch}_iter.tjson"))



if __name__ == "__main__":
    main()