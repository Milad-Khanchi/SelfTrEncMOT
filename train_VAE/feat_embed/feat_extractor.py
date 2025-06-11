import os
import pickle
import cv2
import argparse
from torch.utils.data import DataLoader

from dataset import DanceTrackDataset_test, collate_fn
from app_feat_extractor import app_feat_Extractor


# Argument parser setup
parser = argparse.ArgumentParser(description="Feature extraction script for MOT datasets.")
parser.add_argument("--dataset", type=str, default="dance", choices=["mot17", "mot20", "dance", "sport"], help="Dataset name (default: 'dance')")
parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split (default: 'val')")
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Dataset split (default: 'train')")

args = parser.parse_args()

# Load the DanceTrack dataset
Dataset = args.dataset
split = args.split
mode = args.mode
root_dir = './datasets/'  # Use the provided dataset path

if mode == 'test':
    save_dir = './feat_embed/yoloxd'
else:
    save_dir = './feat_embed'


test_dataset = DanceTrackDataset_test(root_dir=root_dir, split=split, mode=mode)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#Load faeture extractor
feat_extractor = app_feat_Extractor(dataset= Dataset, test= True if split=="test" else False)

data_list = []
for frame_path, frame_bboxes in test_dataloader:
    # Load and preprocess an image.
    img = cv2.imread(frame_path)

    app_feat = feat_extractor.extract_features_app(img, frame_bboxes)

    #Create the file name for saving the features
    components = frame_path.split('/')[-5:]
    components.remove('img1')
    components[-1] = components[-1].replace('.jpg', '.pkl')
    app_path = os.path.join(save_dir, *components)

    if not os.path.exists(app_path):
        # If the file doesn't exist, create it and write
        os.makedirs(os.path.dirname(app_path), exist_ok=True)  # Ensure the directory exists

    # Save the extracted features to a .pkl file
    with open(app_path, 'wb') as f:
        pickle.dump(app_feat, f)

    # print or log the save path
    print(f"Saved apperance features for {frame_path} to {app_path}")

