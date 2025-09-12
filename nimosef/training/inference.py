import os
import time
import torch
import numpy as np
import datetime
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nimosef.models.nimosef import MultiHeadNetwork, SliceCorrection
from nimosef.data.dataset import NiftiDataset, nifti_collate_fn
from nimosef.training.logging import TrainingLogger, LossTracker
from nimosef.training.schedulers import ClampedStepLR
from nimosef.losses.base import dice_loss
from nimosef.losses.composite import CompositeLoss
from nimosef.utils.core import save_checkpoint, label_to_color


def inference_training(args, model=None, split=None, device=None):
    """
    Inference training loop: freeze network, optimize shape codes.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = 4

    hparams = {
        "latent_size": args.latent_size,
        "motion_size": args.motion_size,
        "hidden_size": args.hidden_size,
        "num_res_layers": args.num_res_layers,
        "linear_head": args.linear_head,
        "lr": args.lr_shape_code,
        "lr_shape_code": args.lr_shape_code,
        "lambda_rec": args.lambda_rec,
        "lambda_seg": args.lambda_seg,
        "lambda_reg": args.lambda_reg,
        "lambda_dsp": args.lambda_dsp,
        "lambda_reg_dsp": args.lambda_reg_dsp,
        "lambda_jacobian": args.lambda_jacobian,
        "lambda_vol": args.lambda_vol,
        "lambda_graph_conn": args.lambda_graph_conn,
        "lambda_smoothness": args.lambda_smoothness,
        "warmup_epochs": args.warmup_epochs,
        "max_dsp_weight": args.max_dsp_weight,
    }

    dataset = NiftiDataset(args.split_file, mode=args.mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,        
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        collate_fn=nifti_collate_fn,
    )
    # Setup experiment folders and file names (you may re-use your base save folder)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"inference_{run_id}"    
    base_save_folder = args.save_folder    
    save_folder = os.path.join(base_save_folder, run_name)
    os.makedirs(save_folder, exist_ok=True)
    print(f"Inference results will be saved to {save_folder}")
    
    # Save patient list for validation
    torch.save({'patient_list': dataset.patients}, os.path.join(save_folder, 'validation_patient_list.pth'))
    num_subjects = len(dataset)

    if model is None:
        num_subjects = len(dataset)
        model = MultiHeadNetwork(num_subjects, num_labels,
                                 args.latent_size, args.motion_size,
                                 hidden_size=args.hidden_size,
                                 num_res_layers=args.num_res_layers,
                                 linear_head=args.linear_head)
        
        data_state = torch.load(args.initial_model_path, map_location=device)
        if "model_state_dict" in data_state:
            model_state = data_state["model_state_dict"]
        else:
            model_state = data_state

        # Drop shape codes
        if "shape_code.weight" in model_state:
            del model_state["shape_code.weight"]

        model.load_state_dict(model_state, strict=False)
    
    else:
        # Re-init the shape embeddings
        # Shape codes
        model.shape_code = torch.nn.Embedding(num_subjects, args.latent_size).to(device)
        torch.nn.init.normal_(model.shape_code.weight.data, 0.0, 0.01 / math.sqrt(args.latent_size))

    model.to(device)

    # Freeze everything except shape codes
    for name, param in model.named_parameters():
        if "shape_code" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW([
        {'params': model.shape_code.parameters(), 'lr': args.lr_shape_code},
    ], weight_decay=1e-6)

    scheduler = ClampedStepLR(
        optimizer,
        step_size=args.lr_scheduler_step,
        gamma=args.lr_scheduler_gamma,
        min_lrs=[1e-4, 2e-4],
    )

    writer = SummaryWriter(log_dir=os.path.join(save_folder, "inference_logs"))
    logger = TrainingLogger(log_filename=os.path.join(save_folder, "inference.log"))
    tracker = LossTracker()

    # === Loss ===
    loss_module = CompositeLoss(is_test=True, device=device, hp_dict=hparams)
    scaler = torch.amp.GradScaler(device=device)

    # === Resume from checkpoint ===
    start_epoch = 0
    ckpt_file = getattr(args, "resume_checkpoint", None)
    if ckpt_file and os.path.exists(ckpt_file):
        checkpoint = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        tracker = checkpoint.get("loss_tracker", tracker)
        start_epoch = checkpoint["epoch"]
        print(f"Resumed inference training from {ckpt_file} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.validation_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_metrics = {
            'Total_loss': 0.0,
            'L_intensity': 0.0,
            'L_seg': 0.0,
            'L_latent': 0.0,
            'L_disp': 0.0,
            'L_disp_reg': 0.0,
            'L_J': 0.0,
            'L_Jfold': 0.0,
            "L_Jvol": 0.0,
            'L_graph': 0.0,
            'L_smooth_int': 0.0,
            'L_smooth_seg': 0.0,
            'L_smooth_latent': 0.0,
            'L2_error': 0.0,
            'LV_dice': 0.0,
            'MYO_dice': 0.0,
            'RV_dice': 0.0,            
        }
        epoch_batches = 0

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Inference Epoch {epoch}")):
            optimizer.zero_grad()
            with torch.amp.autocast(device.type):
                # Unpack batch data
                coords, t0_data, tgt_data, t, transform_info, sample_id = batch
                t0_intensity, t0_seg, t0_mask = t0_data
                tgt_intensity, tgt_seg, tgt_mask = tgt_data

                # Sanity checks
                assert tgt_seg.max() < args.num_labels, f"Invalid seg label: {tgt_seg.unique()}"
                assert t0_seg.max() < args.num_labels, f"Invalid t0 seg label: {t0_seg.unique()}"
                assert sample_id.max() < model.shape_code.num_embeddings, f"Sample ID out of range: {sample_id.max()} vs {model.shape_code.num_embeddings}"

                # Send to device
                coords = coords.to(device)
                t0_intensity = t0_intensity.to(device)
                t0_seg = t0_seg.to(device)
                t0_mask = t0_mask.to(device)
                tgt_intensity = tgt_intensity.to(device)
                tgt_seg = tgt_seg.to(device)
                tgt_mask = tgt_mask.to(device)
                t = t.to(device)
                sample_id = sample_id.to(device)

                # Forward passes, t0 (ED or reference time) and t
                t0 = torch.zeros(coords.shape[0], 1, device=device)
                seg_pred_t0, int_pred_t0, disp_t0, _ = model(coords.to(device), t0, sample_id.to(device))

                t = t.unsqueeze(-1).to(device)
                seg_pred_t, int_pred_t, disp_t, _ = model(coords.to(device), t, sample_id.to(device))

                centers = transform_info['centers'][0].to(device)
                affine = transform_info['affines'][0].to(device)
                indices = torch.cat(transform_info['indices'], dim=0).to(device)
                original_resolution = torch.linalg.norm(affine[:3, :3], axis=0) # Voxel resolution

                # ==========================
                # ==== Loss Computation ====
                # ==========================   
                h = model.shape_code(torch.unique(sample_id).to(device))
                preds_t0 = {'seg_pred': seg_pred_t0, 'intensity_pred': int_pred_t0, 'displacement': disp_t0, 'h': h, 'batch_size': args.batch_size}
                preds_t = {'seg_pred': seg_pred_t, 'intensity_pred': int_pred_t, 'displacement': disp_t, 'h': h, 'batch_size': args.batch_size}
                targets_t0 = {'segmentation': t0_seg, 'intensity': t0_intensity, 'mask': t0_mask, 'batch_size': args.batch_size}
                targets_t = {'segmentation': tgt_seg, 'intensity': tgt_intensity, 'mask': tgt_mask, 'batch_size': args.batch_size}

                # Base loss
                total_loss, loss_components = loss_module(sample_id.to(device), coords.to(device), original_resolution,
                                                          preds_t0, preds_t, targets_t0, targets_t, epoch=epoch)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                lv_dice, myo_dice, rv_dice = 1 - dice_loss(seg_pred_t, tgt_seg, num_labels, ignore_background=True, weighted=False, apply_softmax=True)[1:]
                l2_error = ((int_pred_t.squeeze() - tgt_intensity)**2).mean()

                # Update dictionary accumulators:
                epoch_metrics['Total_loss'] += total_loss.item()
                for key, val in loss_components.items():
                    epoch_metrics[key] += val #.item()

                epoch_metrics["LV_dice"] += lv_dice.item()
                epoch_metrics["MYO_dice"] += myo_dice.item()
                epoch_metrics["RV_dice"] += rv_dice.item()
                epoch_metrics['L2_error'] += l2_error.item()

            epoch_batches += 1

        # === End of epoch ===
        for k in epoch_metrics:
            epoch_metrics[k] /= epoch_batches

        tracker.add_loss(epoch_metrics, epoch)
        writer.add_scalar("Inference_Loss/Total_loss", epoch_metrics["Total_loss"], epoch)
        writer.add_scalar("Inference_Dice/LV", epoch_metrics["LV_dice"], epoch)
        writer.add_scalar("Inference_Dice/MYO", epoch_metrics["MYO_dice"], epoch)
        writer.add_scalar("Inference_Dice/RV", epoch_metrics["RV_dice"], epoch)

        # === Save checkpoint ===
        if (epoch + 1) % getattr(args, "checkpoint_epochs", 20) == 0:
            ckpt_path = os.path.join(save_folder, f"inference_checkpoint_last.pth")
            save_checkpoint(ckpt_path, epoch+1, model, optimizer, scheduler, tracker)
        
        # === Print/logging ===
        if (epoch % getattr(args, "print_epochs", 5)) == 0:
            # Log the averages to TensorBoard:
            for key, avg_val in epoch_metrics.items():
                writer.add_scalar(f'Inference_Loss/{key}', avg_val, epoch)

            str_to_print = [f"{key}: {val:.4f}" for key, val in epoch_metrics.items()]
            print(f"Epoch {epoch}: " + ", ".join(str_to_print))

            # - Learning rates
            for i, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Inference_LR/param_group_{i}', param_group['lr'], epoch)
            
            # - Histograms of weights and gradients
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(name, param, epoch)
                    try:
                        if param.grad is not None:
                            writer.add_histogram(f'Inference_Grad/{name}_grad', param.grad, epoch)
                        else:
                            print(f"No gradient for parameter: {name}")
                    except Exception as e:
                        print(f"Error logging gradient for parameter: {name}")
                        print(e)

            # - Predicted images and visualize mid-
            with torch.no_grad():
                # pick one subject ID from the batch
                subject_id = sample_id[0].item()
                idx_subject = (sample_id == subject_id).nonzero(as_tuple=True)[0]

                subj_indices = indices[idx_subject]
                subj_int_pred = int_pred_t0[idx_subject]
                subj_seg_pred = torch.argmax(seg_pred_t0[idx_subject], dim=1)
                subj_t0_intensity = t0_intensity[idx_subject]
                subj_t0_seg = t0_seg[idx_subject]

                # reconstruct shape from max index
                x, y, z = subj_indices.max(0).values.int().tolist()
                x, y, z = x + 1, y + 1, z + 1
                visualize_mid = z // 2

                # allocate full arrays
                pred_im_final = np.zeros((x, y, z))
                pred_seg_final = np.zeros((x, y, z))
                t0_im = np.zeros((x, y, z))
                t0_seg_im = np.zeros((x, y, z))

                # fill arrays using voxel indices
                coords_np = subj_indices.cpu().numpy().astype(np.int16)
                pred_im_final[coords_np[:,0], coords_np[:,1], coords_np[:,2]] = subj_int_pred.squeeze().cpu().numpy()
                pred_seg_final[coords_np[:,0], coords_np[:,1], coords_np[:,2]] = subj_seg_pred.squeeze().cpu().numpy()
                t0_im[coords_np[:,0], coords_np[:,1], coords_np[:,2]] = subj_t0_intensity.squeeze().cpu().numpy()
                t0_seg_im[coords_np[:,0], coords_np[:,1], coords_np[:,2]] = subj_t0_seg.squeeze().cpu().numpy()

                pred_seg_color = label_to_color(pred_seg_final[..., visualize_mid])
                t0_seg_color = label_to_color(t0_seg_im[..., visualize_mid])

                # concat plots
                intensity_plot = np.concatenate((pred_im_final[..., visualize_mid],
                                                t0_im[..., visualize_mid]), axis=-1)
                segmentation_plot = np.concatenate((pred_seg_color, t0_seg_color), axis=1)

            # Convert the label image to a color image:
            writer.add_image('Inference/Reconstruction', intensity_plot, epoch, dataformats='HW')
            writer.add_image('Inference/Segmentation', segmentation_plot.transpose(2, 0, 1), epoch, dataformats='CHW')

            # Log epoch processing time and speed
            epoch_time = time.time() - epoch_start_time
            writer.add_scalar('Inference_Epoch/Time_sec', epoch_time, epoch)
            writer.add_scalar('Inference_Epoch/Batches_per_sec', epoch_batches / epoch_time, epoch)
            print(f"Epoch {epoch} took {epoch_time:.2f} seconds, {epoch_batches/epoch_time:.2f} batches/sec")

        scheduler.step()
        logger.on_epoch_end(epoch, logs=epoch_metrics["Total_loss"])

    # Take as score the average dice of the last epoch
    score = (epoch_metrics['LV_dice'] + epoch_metrics['MYO_dice'] + epoch_metrics['RV_dice']) / 3
    writer.add_hparams(hparams, {'hparam/score': score}, run_name=run_name)
    print(f"Inference finished. Score: {score}")

    # Save the model as validation
    if hasattr(args, 'validation_model_path'):
        validation_model_filename = args.validation_model_path
    else:
        validation_model_filename = None

    if validation_model_filename is None:
        validation_model_filename = os.path.join(save_folder, 'validation_model.pth')
    torch.save(model.state_dict(), validation_model_filename)

    # Save final model
    ckpt_path = os.path.join(save_folder, f"inference_checkpoint_final.pth")
    save_checkpoint(ckpt_path, epoch+1, model, optimizer, scheduler, tracker)
    tracker.save_losses(os.path.join(save_folder, "losses_inference.json"))

    writer.flush()
    writer.close()

    print(f"Validation model saved to {validation_model_filename}")

    return model
