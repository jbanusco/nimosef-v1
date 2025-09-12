import os
import numpy as np
import pandas as pd
import torch
import copy
from nimosef.data.dataset import NiftiDataset
from nimosef.models.nimosef import MultiHeadNetwork
from nimosef.training.args import get_inference_parser



def compute_mean_and_distance(dataset, model, results_folder, overwrite=False):        
    latent_filename = os.path.join(results_folder, "shape_code.parquet")
    distance_filename = os.path.join(results_folder, "shape_code_distances.parquet")

    if os.path.isfile(distance_filename) and not overwrite:
        print(f"Distances already computed!")
        return
    
    # Load state dict    
    shape_code = copy.deepcopy(model.shape_code.weight).detach().cpu().numpy()

    # Create dataframe
    df_shape_code = pd.DataFrame(data=shape_code, index=dataset.patients)

    # Save the latent code    
    df_shape_code.to_parquet(latent_filename)

    # Save the L2 distance to the mean shape code
    mean_latent = np.mean(shape_code, axis=0)
    distances = np.linalg.norm(shape_code - mean_latent, axis=1)

    # Save the distances
    df_distances = pd.DataFrame(data=distances, index=dataset.patients, columns=['Distance'])    
    df_distances.to_parquet(distance_filename)

    print(f'Saved shape code to {distance_filename}')


def main():
    parser = get_inference_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dataset ===
    dataset = NiftiDataset(args.split_file, mode=args.mode)

    # === Model ===
    model = MultiHeadNetwork(
        num_subjects=len(dataset),
        num_labels=args.num_labels,
        latent_size=args.latent_size,
        motion_size=args.motion_size,
        hidden_size=args.hidden_size,
        num_res_layers=args.num_res_layers,
        linear_head=args.linear_head,
    )
    model.to(device)

    # === Load pretrained weights ===
    ckpt = torch.load(args.model_to_rec, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt        
    model.load_state_dict(state_dict, strict=True)    

    results_folder = os.path.join(args.save_rec_folders, f"results_{args.mode}_comparison")
    compute_mean_and_distance(dataset, model, results_folder, args.overwrite_imgs)


if __name__ == "__main__":
    main()
