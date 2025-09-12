import os
from nimosef.training.train import train_model
from nimosef.training.args import get_train_parser

def main():
    data_path = os.environ.get("DATA_PATH", "/media/jaume/DATA/Data/Test_NIMOSEF_Dataset")
    manifest_file = os.path.join(data_path, "derivatives", "manifests_nimosef", "dataset_manifest.json")
    save_folder = os.path.join(data_path, "derivatives", "nimosef")
    previous_checkpoint = os.path.join(data_path, "derivatives/nimosef/experiment_20250909_153354/checkpoint_final.pth")
    model_path = os.path.join(data_path, "derivatives/nimosef/experiment_20250909_104844/model.pth")

    # Load parser with defaults
    parser = get_train_parser()
    args = parser.parse_args([
        "--data_folder", data_path,
        "--split_file", manifest_file,
        "--save_folder", save_folder,
        "--checkpoint_filename", previous_checkpoint,
        "--resume_from_checkpoint", "False",
        "--load_model", "True",
        "--model_path", model_path,
        "--batch_size", "1",
        "--prefetch_factor", "2",
        "--num_workers", "4",
        "--num_epochs", "3000",
        "--epochs_to_evaluate_validation", "1000",
        "--validation_epochs", "50",
        "--lambda_rec", "5.0",
        "--lambda_seg", "2.0",
        "--lambda_reg", "0.01",
        "--lambda_dsp", "2.0",
        "--lambda_reg_dsp", "1.0",
        "--lambda_jacobian", "5.0",
        "--lambda_vol", "0.01",
        "--lambda_graph_conn", "0.0",
        "--lambda_smoothness", "0.5",
        "--warmup_epochs", "50",
        "--lr", "0.0005",
        "--lr_shape_code", "0.001",
        "--lr_scheduler_step", "10",
        "--lr_scheduler_gamma", "0.95",
    ])
    print(">>> Starting training on", args.data_folder)
    train_model(args)

    # Save symlink to last model for inference
    import glob, shutil
    run_folders = sorted(glob.glob(os.path.join(save_folder, "experiment_*")))
    if run_folders:
        final_model = os.path.join(run_folders[-1], "model.pth")
        link_path = os.path.join(save_folder, "latest_model.pth")
        if os.path.exists(final_model):
            shutil.copy(final_model, link_path)
            print(f">>> Copied latest model to {link_path}")

if __name__ == "__main__":
    main()
