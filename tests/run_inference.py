import os
from nimosef.training.inference import inference_training
from nimosef.training.args import get_inference_parser

def main():
    data_path = os.environ.get("DATA_PATH", "/media/jaume/DATA/Data/Test_NIMOSEF_Dataset")
    manifest_file = os.path.join(data_path, "derivatives", "manifests_nimosef", "dataset_manifest.json")
    save_folder = os.path.join(data_path, "derivatives", "nimosef_v1")
    initial_model_path = os.path.join(save_folder, "experiment_20250907_183826", "checkpoint_last.pth")

    # Load parser with defaults
    parser = get_inference_parser()
    args = parser.parse_args([
        "--data_folder", data_path,
        "--split_file", manifest_file,
        "--save_folder", save_folder,
        "--initial_model_path", initial_model_path,
    ])

    print(">>> Starting inference using model:", args.initial_model_path)
    inference_training(args)

if __name__ == "__main__":
    main()
