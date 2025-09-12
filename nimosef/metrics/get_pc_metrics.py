import os
import numpy as np
import pandas as pd
import argparse
import torch

from nimosef.data.dataset import NiftiDataset
from nimosef.metrics.pc_metrics import compute_distance_pts


def get_pc_metrics(args):
    # Prepare the dataset
    dataset = NiftiDataset(args.splits_filename, mode=args.mode)

    # Final results folder
    save_folder_results = os.path.join(args.results_folder, f"results_{args.mode}_comparison")
    os.makedirs(save_folder_results, exist_ok=True)

    # Get the list of subjects in the baseline data
    num_subjects = len(dataset.patients)

    list_hf95 = []
    list_hf = []
    list_chamfer_dis = []

    for idx in range(0, num_subjects):
        print(f'Processing subject {idx+1}/{num_subjects}')
        subject_id = dataset.patients[idx]
            
        print(f"Subject: {subject_id}")    
        path_chamfer_subj = os.path.join(args.results_folder, subject_id, f'{subject_id}_chamfer.parquet')
        path_hf95_subj = os.path.join(args.results_folder, subject_id, f'{subject_id}_hf95.parquet')
        path_hf_subj = os.path.join(args.results_folder, subject_id, f'{subject_id}_hf.parquet')

        do_pc = True
        if os.path.isfile(path_chamfer_subj) and os.path.isfile(path_hf95_subj) and os.path.isfile(path_hf_subj):
            do_pc = False
        
        if args.force_overwrite:
            do_pc = True

        if do_pc:
            pred_boundaries_path = os.path.join(args.results_folder, subject_id, f'{subject_id}_pred_boundaries.parquet')
            pred_boundaries = pd.read_parquet(pred_boundaries_path)            

            true_boundaries_path = os.path.join(args.results_folder, subject_id, f'{subject_id}_true_boundaries.parquet')
            true_boundaries = pd.read_parquet(true_boundaries_path)

            scale_factor = np.array(dataset.bbox) / 2
            pred_boundaries['x'] = pred_boundaries['x'] * scale_factor[0]
            pred_boundaries['y'] = pred_boundaries['y'] * scale_factor[1]
            pred_boundaries['z'] = pred_boundaries['z'] * scale_factor[2]            

            true_boundaries['x'] = true_boundaries['x'] * scale_factor[0]
            true_boundaries['y'] = true_boundaries['y'] * scale_factor[1]
            true_boundaries['z'] = true_boundaries['z'] * scale_factor[2]

            frames_ids = np.unique(pred_boundaries['time'])
            num_frames = len(frames_ids)

            list_chamfer_dis_subj = []
            list_hf95_subj = []
            list_hf_subj = []
            for time_idx in range(0, num_frames):
                pts1 = pred_boundaries.query(f'time == {time_idx}')[['x', 'y', 'z']].values
                pts2 = true_boundaries.query(f'time == {time_idx}')[['x', 'y', 'z']].values

                # Assume pts1 and pts2 are tensors of shape (B, N, 3) and (B, M, 3)
                pts1 = torch.tensor(pts1).unsqueeze(0)
                pts2 = torch.tensor(pts2).unsqueeze(0)

                chamfer_dis = compute_distance_pts(pts1, pts2, percentile=1.0, chamfer=True)    
                hf95 = compute_distance_pts(pts1, pts2, percentile=0.95)        
                hf = compute_distance_pts(pts1, pts2, percentile=1.0)    

                list_chamfer_dis_subj.append(chamfer_dis)
                list_hf95_subj.append(hf95)
                list_hf_subj.append(hf)

            chamfer_distances = np.array(list_chamfer_dis_subj)
            hf95_distances = np.array(list_hf95_subj)
            hf_distances = np.array(list_hf_subj)

            df_chamfer_subj = pd.DataFrame(data=chamfer_distances, columns=[subject_id]).T
            df_hf95_subj = pd.DataFrame(data=hf95_distances, columns=[subject_id]).T
            df_hf_subj = pd.DataFrame(data=hf_distances, columns=[subject_id]).T

            df_chamfer_subj.to_parquet(path_chamfer_subj)
            df_hf95_subj.to_parquet(path_hf95_subj)
            df_hf_subj.to_parquet(path_hf_subj)
        else:
            df_chamfer_subj = pd.read_parquet(path_chamfer_subj)
            df_hf95_subj = pd.read_parquet(path_hf95_subj)
            df_hf_subj = pd.read_parquet(path_hf_subj)


        list_chamfer_dis.append(df_chamfer_subj)
        list_hf95.append(df_hf95_subj)
        list_hf.append(df_hf_subj)

    chamfer_distances = pd.concat(list_chamfer_dis, ignore_index=False)
    hf95_distances = pd.concat(list_hf95, ignore_index=False)
    hf_distances = pd.concat(list_hf, ignore_index=False)

    path_chamfer = os.path.join(save_folder_results, 'chamfer.parquet')
    path_hf95 = os.path.join(save_folder_results, 'hf95.parquet')
    path_hf = os.path.join(save_folder_results, 'hf.parquet')
    chamfer_distances.to_parquet(path_chamfer)
    hf95_distances.to_parquet(path_hf95)
    hf_distances.to_parquet(path_hf)



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
    get_pc_metrics(args)


if __name__ == '__main__':
    main()