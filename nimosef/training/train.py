import os
import time
import copy
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from nimosef.models.nimosef import MultiHeadNetwork
from nimosef.data.dataset import NiftiDataset, nifti_collate_fn
from nimosef.training.logging import TrainingLogger, LossTracker
from nimosef.training.schedulers import ClampedStepLR
from nimosef.training.inference import inference_training

from nimosef.utils.core import save_checkpoint, label_to_color
from nimosef.losses.composite import CompositeLoss
from nimosef.losses.base import dice_loss


def train_model(args):
    """
    Main training loop for NIMOSEF.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = 4

    # === Dataset ===
    dataset = NiftiDataset(args.split_file, mode=args.mode)
    if args.debug:
        subset_indices = list(range(min(len(dataset), 10)))
        dataset = Subset(dataset, subset_indices)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,        
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        collate_fn=nifti_collate_fn,
    )

    # === Model ===
    num_subjects = len(dataset)
    model = MultiHeadNetwork(
        num_subjects, num_labels,
        latent_size=args.latent_size,
        motion_size=args.motion_size,
        hidden_size=args.hidden_size,
        num_res_layers=args.num_res_layers,
        linear_head=args.linear_head,
    )
    model.to(device)

    # === Optimizer & Scheduler ===
    optimizer = torch.optim.AdamW([
        {'params': [p for n,p in model.named_parameters() 
                    if all(k not in n for k in ['shape_code']) and p.requires_grad], 
        'lr': args.lr},
        {'params': model.shape_code.parameters(), 'lr': args.lr_shape_code},
    ], weight_decay=1e-6)

    scheduler = ClampedStepLR(
        optimizer,
        step_size=args.lr_scheduler_step,
        gamma=args.lr_scheduler_gamma,
        min_lrs=[1e-4, 2e-4],
    )

    # === Logging ===
    run_name = time.strftime("experiment_%Y%m%d_%H%M%S")
    save_folder = os.path.join(args.save_folder, run_name)
    os.makedirs(save_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_folder, 'logs'))
    logger = TrainingLogger(log_filename=os.path.join(save_folder, 'train.log'))
    tracker = LossTracker()

    # Log hyperparameters
    hparams = {
        "latent_size": args.latent_size,
        "motion_size": args.motion_size,
        "hidden_size": args.hidden_size,
        "num_res_layers": args.num_res_layers,
        "linear_head": args.linear_head,
        "lr": args.lr,
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
    with open(os.path.join(save_folder, "hyperparameters.json"), "w") as f:
        json.dump(hparams, f, indent=4)
    writer.add_hparams(hparams, {"hparam/score": 0.0}, run_name=run_name)

    # === Loss ===
    loss_module = CompositeLoss(is_test=False, device=device, hp_dict=hparams)
    scaler = torch.amp.GradScaler(device=device)

    # === Resume from checkpoint ===
    start_epoch = 0
    if getattr(args, "resume_from_checkpoint", False) and getattr(args, "checkpoint_filename", None):
        ckpt_path = os.path.join(args.save_folder, args.checkpoint_filename)
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            tracker = checkpoint.get("loss_tracker", tracker)
            start_epoch = checkpoint["epoch"]
            print(f"Resumed training from checkpoint {ckpt_path} at epoch {start_epoch}")
        else:
            print(f"⚠️ Checkpoint {ckpt_path} not found, starting fresh.")

    # === Load previous model ===
    if getattr(args, "load_model", False) and getattr(args, "model_path", None):
        model_path = args.model_path
        if os.path.exists(model_path):
            model_state = torch.load(model_path, map_location=device)
            if "shape_code.weight" in model_state:
                del model_state["shape_code.weight"]
            model.load_state_dict(model_state, strict=False)

            # No need to re-start the embeddings since they are already initialized randomly.

            print(f"Load model from {model_path}")
        else:
            print(f"⚠️ Model {model_path} not found.")

    # === Epoch loop ===
    for epoch in range(start_epoch, args.num_epochs):
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
        valid_jacobian = False

        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
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

                # Sample a percentage of the data
                num_points = coords.shape[0]
                num_train_points = int(args.sample_percentage * num_points)                
                idx_points = torch.tensor(np.random.choice(num_points, num_train_points, replace=False)).long()

                coords = coords[idx_points].to(device)
                t0_intensity = t0_intensity[idx_points].to(device)
                t0_seg = t0_seg[idx_points].to(device)
                t0_mask = t0_mask[idx_points].to(device)
                tgt_intensity = tgt_intensity[idx_points].to(device)
                tgt_seg = tgt_seg[idx_points].to(device)
                tgt_mask = tgt_mask[idx_points].to(device)
                t = t[idx_points].to(device)
                sample_id = sample_id[idx_points].to(device)

                # Forward passes, t0 (ED or reference time) and t
                t0 = torch.zeros(coords.shape[0], 1, device=device)
                seg_pred_t0, int_pred_t0, disp_t0, _ = model(coords.to(device), t0, sample_id.to(device))

                t = t.unsqueeze(-1).to(device)
                seg_pred_t, int_pred_t, disp_t, _ = model(coords.to(device), t, sample_id.to(device))

                # Get transform info for rescaling
                centers = transform_info['centers'][0].to(device)
                affine = transform_info['affines'][0].to(device)
                indices = torch.cat(transform_info['indices'], dim=0).to(device)
                indices = indices[idx_points]

                original_resolution = torch.linalg.norm(affine[:3, :3], axis=0) # Voxel resolution

                # ==========================
                # ==== Loss Computation ====
                # ==========================
                h = model.shape_code(torch.unique(sample_id))
                preds_t0 = {'seg_pred': seg_pred_t0, 'intensity_pred': int_pred_t0, 'displacement': disp_t0, 'h': h, 'batch_size': args.batch_size}
                preds_t = {'seg_pred': seg_pred_t, 'intensity_pred': int_pred_t, 'displacement': disp_t, 'h': h, 'batch_size': args.batch_size}
                targets_t0 = {'segmentation': t0_seg, 'intensity': t0_intensity, 'mask': t0_mask, 'batch_size': args.batch_size}
                targets_t = {'segmentation': tgt_seg, 'intensity': tgt_intensity, 'mask': tgt_mask, 'batch_size': args.batch_size}

                # Base loss
                total_loss, loss_components = loss_module(sample_id.to(device), coords.to(device), original_resolution,
                                                          preds_t0, preds_t, targets_t0, targets_t, epoch=epoch)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Track metrics
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
        writer.add_scalar("Loss/Total_loss", epoch_metrics["Total_loss"], epoch)
        writer.add_scalar("Dice/LV", epoch_metrics["LV_dice"], epoch)
        writer.add_scalar("Dice/MYO", epoch_metrics["MYO_dice"], epoch)
        writer.add_scalar("Dice/RV", epoch_metrics["RV_dice"], epoch)

        scheduler.step()
        logger.on_epoch_end(epoch, logs=epoch_metrics["Total_loss"])

        # Optional validation
        if (epoch + 1) % args.epochs_to_evaluate_validation == 0:
            ckpt_path = os.path.join(save_folder, f"checkpoint_{epoch+1}.pth")  # Since here we evaluate inference
            save_checkpoint(ckpt_path, epoch+1, model, optimizer, scheduler, tracker)
            inference_training(args, copy.deepcopy(model), split="val", device=device)

        # === Save checkpoint ===
        if (epoch + 1) % getattr(args, "checkpoint_epochs", 20) == 0:
            # Overwrite previous one
            ckpt_path = os.path.join(save_folder, f"checkpoint_last.pth")
            save_checkpoint(ckpt_path, epoch+1, model, optimizer, scheduler, tracker)

        # === Print/logging ===
        if (epoch % getattr(args, "print_epochs", 5)) == 0:
            # Log the averages to TensorBoard:
            for key, avg_val in epoch_metrics.items():
                writer.add_scalar(f'Losses/{key}', avg_val, epoch)

            str_to_print = [f"{key}: {val:.4f}" for key, val in epoch_metrics.items()]
            print(f"Epoch {epoch}: " + ", ".join(str_to_print))

            # - Learning rates
            for i, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'LR/param_group_{i}', param_group['lr'], epoch)
            
            # - Histograms of weights and gradients
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(name, param, epoch)
                    try:
                        if param.grad is not None:
                            writer.add_histogram(f'{name}_grad', param.grad, epoch)
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
            writer.add_image('Reconstruction', intensity_plot, epoch, dataformats='HW')
            writer.add_image('Segmentation', segmentation_plot.transpose(2, 0, 1), epoch, dataformats='CHW')

            # Log epoch processing time and speed
            epoch_time = time.time() - epoch_start_time
            writer.add_scalar('Epoch/Time_sec', epoch_time, epoch)
            writer.add_scalar('Epoch/Batches_per_sec', epoch_batches / epoch_time, epoch)
            print(f"Epoch {epoch} took {epoch_time:.2f} seconds, {epoch_batches/epoch_time:.2f} batches/sec")

        scheduler.step()  # Update learning rate
        logger.on_epoch_end(epoch, logs=total_loss.detach().item())

    # Take as score the average dice of the last epoch
    score = (epoch_metrics['LV_dice'] + epoch_metrics['MYO_dice'] + epoch_metrics['RV_dice']) / 3
    writer.add_hparams(hparams, {'hparam/score': score}, run_name=run_name)
    print(f"Training finished. Score: {score}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_folder, "model.pth"))  # Just the model
    ckpt_path = os.path.join(save_folder, f"checkpoint_final.pth")
    save_checkpoint(ckpt_path, epoch+1, model, optimizer, scheduler, tracker)
    tracker.save_losses(os.path.join(save_folder, "losses.json"))
    writer.flush()
    writer.close()
