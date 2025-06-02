import os
import pickle
import sys
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from util import *
from dotenv import load_dotenv
load_dotenv()


from dataset import DanceTrackDataset_test, collate_fn

sam2_path = os.getenv("SAM2_PATH")
if sam2_path:
    sys.path.append(sam2_path)
    from sam2.build_sam import build_sam2_video_predictor
else:
    raise EnvironmentError("SAM2_PATH is not set in .env file")
# Argument parser setup
parser = argparse.ArgumentParser(description="Feature extraction script for MOT datasets.")
parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split (default: 'val')")
parser.add_argument("--mode", type=str, default="val", choices=["train", "test"], help="Dataset split (default: 'train')")

args = parser.parse_args()

# Load the DanceTrack dataset
split = args.split
mode = args.mode
root_dir = os.getenv("DATASET_PATH", "datasets")  # fallback to "datasets" if not set

if mode == 'test':
    save_dir = './train_VAE/prompt_mask/yoloxd'
else:
    save_dir = './train_VAE/prompt_mask'


test_dataset = DanceTrackDataset_test(root_dir=root_dir, split=split, mode=mode)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

# Load model
sam2_checkpoint = os.getenv("SAM2_CHECKPOINT")
model_cfg = os.getenv("MODEL_CFG")

model = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')

transform_output = transforms.Compose([
    transforms.ToPILImage(),              # Convert tensor to PIL image
    transforms.Resize((256, 256)),         # Resize the image
    transforms.ToTensor(),                # Convert back to tensor
])

for frame_path, frame_bboxes, bboxes in test_dataloader:

    inference_state = model.init_state(video_path=frame_path)
    model.reset_state(inference_state)
    for i in range(len(frame_bboxes)):
        _, out_obj_ids, out_mask_logits = model.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=i,
            box=frame_bboxes[i],
        )
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: out_mask_logits[i]
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    embs0 = [(element).squeeze(0) for element in
             video_segments[0].values()]
    #embs0 = torch.stack([F.interpolate(eb.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0).squeeze(0) for eb in embs0])
    embs0 = torch.stack(embs0)
    # embs0 = torch.nn.functional.normalize(embs0, dim=-1)
    embs1 = [(element).squeeze(0) for element in
             video_segments[1].values()]
    embs1 = torch.stack(embs1)
    #embs1 = torch.stack([F.interpolate(eb.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0).squeeze(0) for eb in embs1])
    # embs1 = torch.nn.functional.normalize(embs1, dim=-1)
    # masked_out = (embs0.cpu().numpy(), embs1.cpu().numpy())

    # # Plot the mask
    # plt.imshow((embs1[1] > 0).cpu().numpy())  # Use 'gray' colormap for binary masks
    # plt.title("Binary Mask")
    # plt.axis('off')  # Hide axes for better visualization
    # plt.savefig(f'./prompt_mask/test_mask_one.png')
    # plt.show()
    # plt.close()

    masked_out = []
    bboxes = np.array(bboxes)
    for i in range(len(bboxes)):
        masked_out.append(((embs0[i] > 0).cpu().to(torch.bool), (embs1[i] > 0).cpu().to(torch.bool), bboxes[i]))

    # #plot masks
    # plt.close("all")
    # for out_frame_idx in range(0, len(frame_path)):
    #     plt.figure(figsize=(6, 4))
    #     plt.title(f"frame {out_frame_idx}")
    #     plt.imshow(Image.open(frame_path[out_frame_idx]))
    #     if out_frame_idx == 0:
    #         for bxr in range(frame_bboxes.shape[0]):
    #             show_box(frame_bboxes[bxr], plt.gca())
    #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    #
    #     plt.savefig(f'./prompt_mask/test_mask{out_frame_idx}.png')
    #     plt.close("all")
    # #until here

    # Create the file name for saving the features
    components = frame_path[0].split('/')[-5:]
    components.remove('img1')
    components[-1] = components[-1].replace('.jpg', '.pkl')
    depth_path = os.path.join(save_dir, *components)

    if not os.path.exists(depth_path):
        # If the file doesn't exist, create it and write
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)  # Ensure the directory exists

    # Save the extracted features to a .pkl file
    with open(depth_path, 'wb') as f:
        pickle.dump(masked_out, f)

    # print or log the save path
    print(f"Saved depth features for {frame_path} to {depth_path}\n")

print('Done!')


