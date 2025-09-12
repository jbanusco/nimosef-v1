import os
import numpy as np
import pandas as pd
import argparse

from nimosef.data.dataset import NiftiDataset
from nimosef.utils.visualization import get_imgs_experiment, compute_label_volumes
from nimosef.metrics.dice import dice_coefficient
from nimosef.metrics.img_metrics import count_connected_components_3d_df


def get_img_metrics(args):
    # Prepare the dataset
    dataset = NiftiDataset(args.splits_filename, mode=args.mode)

    # Final results folder
    save_folder_results = os.path.join(args.results_folder, f"results_{args.mode}_comparison")
    os.makedirs(save_folder_results, exist_ok=True)

    # Get the list of subjects in the baseline data
    subjects = dataset.patients
    num_subjects = len(subjects)

    df_volumes = pd.DataFrame()  # Will store long-format DataFrame for each subject/segmentation type
    final_df = pd.DataFrame()  # Will store the final DataFrame with all subjects
    df_dice_final = pd.DataFrame()

    for idx in range(0, num_subjects):
        print(f'Processing subject {idx+1}/{num_subjects}')
        subject_id = subjects[idx]

        summary_path_subj = os.path.join(args.results_folder, subject_id, f'{subject_id}_connected_components.parquet')        
        path_to_volume_subj = os.path.join(args.results_folder, subject_id, f'{subject_id}_volume.parquet')        
        path_to_dice_subj = os.path.join(args.results_folder, subject_id, f'{subject_id}_dice.parquet')
        
        do_volume = False if os.path.isfile(path_to_volume_subj) else True
        do_cc = False if os.path.isfile(summary_path_subj) else True
        do_dice = False if os.path.isfile(path_to_dice_subj) else True
        if args.force_overwrite:
            do_volume = True
            do_cc = True
            do_dice = True

        # Change this to the desired slice index, or set to None for full volume
        z_slice = None 
        if do_cc or do_volume or do_dice:
            im_gt, im_pred, seg_gt, seg_pred, dvol  = get_imgs_experiment(args.results_folder, subject_id)
            labels = np.unique(seg_gt)
            labels = [label for label in labels if label != 0]

        # ========================= Dice
        if do_dice:            
            dice_global = dice_coefficient(seg_pred, seg_gt, smooth=1e-6, per_slice=False, per_frame=False, apply_softmax=False)            

            df_dice_subject = pd.DataFrame(data=dice_global, columns=[subject_id])        
            df_dice_subject = df_dice_subject.T 
            df_dice_subject.columns = ['LV', 'MYO', 'RV']
            df_dice_subject = df_dice_subject.astype(float)
            df_dice_subject.index.name = 'Subject'
            df_dice_subject.reset_index(inplace=True, drop=False)
            df_dice_subject['segmentation_type'] = "pred_base"        
            df_dice_subject.to_parquet(path_to_dice_subj)
        else:
            df_dice_subject = pd.read_parquet(path_to_dice_subj)        
        df_dice_final = pd.concat([df_dice_final, df_dice_subject], ignore_index=True)

        # ======================= Volumes
        if do_volume:
            results_subject = []
                        
            volumes_gt = compute_label_volumes(seg_gt, labels, z_slice=z_slice, dvol=dvol, to_ml=True)
            volumes_pred = compute_label_volumes(seg_pred, labels, z_slice=z_slice, dvol=dvol, to_ml=True)            

            df_gt = pd.DataFrame(volumes_gt)
            df_gt.columns = ['LV', 'Myo', 'RV']
            df_gt.index.name = 'Time'

            df_pred = pd.DataFrame(volumes_pred)
            df_pred.columns = ['LV', 'Myo', 'RV']
            df_pred.index.name = 'Time'

            # Compute the derivative using the diff method (difference between consecutive time points)
            df_gt_deriv = df_gt.apply(np.gradient)  # np.gradient(series) computes the derivative using central differences
            df_pred_deriv = df_pred.apply(np.gradient)

            # Create a long-format DataFrame for each segmentation type
            for seg_type, df_vol, df_deriv in [('gt', df_gt, df_gt_deriv), ('pred', df_pred, df_pred_deriv)]:
                # Convert wide to long format for volumes and derivatives
                df_vol_long = df_vol.reset_index().melt(id_vars='Time', var_name='Class', value_name='Volume')
                df_deriv_long = df_deriv.reset_index().melt(id_vars='Time', var_name='Class', value_name='Derivative')
                # Merge volume and derivative information on Time and Class
                df_merged = pd.merge(df_vol_long, df_deriv_long, on=['Time', 'Class'])
                # Add subject id and segmentation type
                df_merged['Subject'] = subject_id
                df_merged['segmentation_type'] = seg_type
                results_subject.append(df_merged)
            
            df_volume_subject = pd.concat(results_subject, ignore_index=True)
            df_volume_subject.to_parquet(path_to_volume_subj)
        else:
            df_volume_subject = pd.read_parquet(path_to_volume_subj)
        df_volumes = pd.concat([df_volumes, df_volume_subject], ignore_index=True)

        # ======================= Connected components
        if do_cc:                        
            df_gt = count_connected_components_3d_df(seg_gt, background=0, time_axis=-1)
            df_pred = count_connected_components_3d_df(seg_pred, background=0, time_axis=-1)            

            # Add a column to differentiate segmentation types.
            df_gt["segmentation_type"] = "gt"
            df_pred["segmentation_type"] = "pred"            

            # Also add the subject ID.
            df_gt["Subject"] = subject_id
            df_pred["Subject"] = subject_id            

            # Concatenate the DataFrames into one.
            combined_df = pd.concat([df_gt, df_pred], ignore_index=True)

            # Now, group by subject, segmentation type, and label, and compute both the average and maximum
            # number of connected components for each label.
            summary_df = (
                combined_df
                .groupby(["Subject", "segmentation_type", "label"])["component_count"]
                .agg(avg_components="mean", max_components="max")
                .reset_index()
            )

            summary_df.to_parquet(summary_path_subj)            
        else:
            summary_df = pd.read_parquet(summary_path_subj)

        final_df = pd.concat([final_df, summary_df], ignore_index=True)    

    volumes_df_path = os.path.join(save_folder_results, 'volumes_imgs.parquet')
    df_volumes.to_parquet(volumes_df_path)

    final_df_path = os.path.join(save_folder_results, 'connected_components.parquet')
    final_df.to_parquet(final_df_path)

    dice_df_path = os.path.join(save_folder_results, 'dice_score.parquet')
    df_dice_final.to_parquet(dice_df_path)


def get_parser():
    data_path="/media/jaume/DATA/Data/Test_NIMOSEF_Dataset"
    splits_filename=f"{data_path}/derivatives/manifests_nimosef/dataset_manifest.json"    

    mode="train"
    results_folder = f"{data_path}/derivatives/nimosef_results"

    parser = argparse.ArgumentParser(description="Generate train/val/test split for NiftiDataset")
    parser.add_argument("--data_path", type=str, default=data_path)
    parser.add_argument("--splits_filename", type=str, default=splits_filename)
    parser.add_argument("--mode", type=str, default=mode)
    parser.add_argument("--results_folder", type=str, default=results_folder)
    parser.add_argument("--force_overwrite", action="store_true", help="Force overwrite existing files")
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    # args.force_overwrite = True
    get_img_metrics(args)


if __name__ == '__main__':
    main()
