import os
import numpy as np
import pandas as pd
import torch
import copy
from nimosef.data.dataset import NiftiDataset
from nimosef.models.nimosef import MultiHeadNetwork
from nimosef.training.args import get_inference_parser
from nimosef.training.generate_results import generate_reconstructed_images


def generate_mean_img(dataset, model, results_folder, res_factor_z, overwrite=False):
    # The model    
    model.eval()

    # Load the mean shape code
    latent_filename = os.path.join(results_folder, "shape_code.parquet")
    shape_code = pd.read_parquet(latent_filename)
    mean_latent = shape_code.mean(axis=0).values[None, ...]  # Mean/average latent space

    model.shape_code = torch.nn.Embedding(1, model.latent_size).requires_grad_(False)
    model.shape_code.weight = torch.nn.Parameter(torch.tensor(mean_latent), requires_grad=False)    

    # No motion correction        
    generate_reconstructed_images(model, dataset, results_folder, res_factor_z=res_factor_z, 
                                  reprocess=overwrite, num_subjects=len(dataset.patients), is_synthetic=True,
                                  save_motion_corrected=False)


def main():
    parser = get_inference_parser()
    args = parser.parse_args()

    # === Dataset ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NiftiDataset(args.split_file, mode=args.mode)
    dataset.patients = ["mean"]

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

    # Remove the shape code
    del state_dict['shape_code.weight']

    # Load without shape code embeddings
    model.load_state_dict(state_dict, strict=False)

    results_folder = os.path.join(args.save_rec_folders, f"results_{args.mode}_comparison")
    generate_mean_img(dataset, model, results_folder, args.res_factor_z, overwrite=args.overwrite_imgs)


if __name__ == "__main__":
    main()
