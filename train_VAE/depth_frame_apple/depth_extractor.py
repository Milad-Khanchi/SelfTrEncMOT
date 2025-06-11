import os
import pickle
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader

from dataset import DanceTrackDataset_test, collate_fn
import depth_pro

from dotenv import load_dotenv
load_dotenv()

# Argument parser setup
parser = argparse.ArgumentParser(description="Feature extraction script for MOT datasets.")
parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split (default: 'val')")
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Dataset split (default: 'train')")

args = parser.parse_args()

load_dotenv()
root_dir = os.getenv("DATASET_PATH", "datasets")  # fallback to "datasets" if not set

# Load the DanceTrack dataset
split = args.split
mode = args.mode

save_dir = '/train_VAE/depth_frame_apple'


test_dataset = DanceTrackDataset_test(root_dir=root_dir, split=split, mode=mode)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

data_list = []
for frame_path, frame_bboxes in test_dataloader:
    # Load and preprocess an image.
    image, _, f_px = depth_pro.load_rgb(frame_path)
    img = transform(image)
    original_size = image.shape[:-1]

    # Generate depth maps
    prediction = model.infer(img, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].

    # # plot
    # crop1 = depth.cpu().numpy()
    # plt.imshow(crop1)
    # plt.axis('off')  # Hide axes for better visualization
    # plt.savefig(f'/depth_frame_apple/test_image_pairs_batch_{1}.png')
    # plt.show()
    # # until here

    depth_feat = (depth.cpu().unsqueeze(0))

    #Create the file name for saving the features
    components = frame_path.split('/')[-5:]
    components.remove('img1')
    components[-1] = components[-1].replace('.jpg', '.pkl')
    depth_path = os.path.join(save_dir, *components)

    if not os.path.exists(depth_path):
        # If the file doesn't exist, create it and write
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)  # Ensure the directory exists

    # Save the extracted features to a .pkl file
    with open(depth_path, 'wb') as f:
        pickle.dump(depth_feat, f)

    # print or log the save path
    print(f"Saved depth features for {frame_path} to {depth_path}\n")

print('Done!')
