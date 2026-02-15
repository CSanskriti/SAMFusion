import yaml
from opencood.data_utils.datasets.opv2v.intermediate_fusion_dataset import IntermediateFusionDataset
from torch.utils.data import DataLoader

# --- Load your YAML config (same one used in training) ---
cfg_path = "hypes_yaml/V2X-R/L_4DR_Fusion_with_MDD/V2XR_AttFuse_with_MDD.yaml"
with open(cfg_path, 'r') as f:
    hypes = yaml.safe_load(f)

# --- Override dataset root directory to your local path ---
hypes['root_dir'] = "/home/cse/Downloads/RL_3DOD/dataset_decompress"

# --- Create the validation dataset ---
val_dataset = IntermediateFusionDataset(hypes, visualize=False, train=False)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Iterate once and print timestamps ---
all_timestamps = []

for idx in range(len(val_dataset)):
    data = val_dataset.retrieve_base_data(idx, cur_ego_pose_flag=False)
    cavs = data[0]  # base_data_dict
    # Collect all timestamp keys from the first CAV
    ts_keys = list(list(cavs.values())[0].keys())
    print(f"Index {idx} -> timestamps: {ts_keys}")
    all_timestamps.extend(ts_keys)

# Save to a file if needed
with open("val_timestamps.txt", "w") as f:
    for ts in all_timestamps:
        f.write(ts + "\n")

print(f"Collected {len(all_timestamps)} timestamps in total.")
