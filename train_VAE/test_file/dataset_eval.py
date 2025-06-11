import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
from torchvision import transforms
import numpy as np

from dotenv import load_dotenv
load_dotenv()

transform_depth = transforms.Compose([
    transforms.Lambda(lambda x: torch.nn.functional.interpolate(
        x, size=(128, 128), mode='bicubic', align_corners=False)),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    # transforms.Lambda(lambda x: x.half())  # Convert to float16
])

RGB_trs = transforms.Compose([
    transforms.ToTensor(),
])

class DanceTrackDataset(Dataset):
    def __init__(self, root_dir, depth_feat_dir, app_feat_dir, mask_feat_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.depth_feat_dir = depth_feat_dir
        self.app_feat_dir = app_feat_dir
        self.mask_feat_dir = mask_feat_dir
        self.split = split
        self.transform = transform
        self.depth_transform = transform_depth
        self.data_info = self._load_data_info()
        self.RGB_trf = RGB_trs

    def _load_data_info(self):
        data_info = []
        split_dir = os.path.join(self.root_dir, 'DANCETRACK', self.split)
        depth_dir = os.path.join(self.depth_feat_dir, 'DANCETRACK', self.split)
        app_dir = os.path.join(self.app_feat_dir, 'yoloxd/DANCETRACK', self.split)
        mask_dir = os.path.join(self.mask_feat_dir, 'yoloxd/DANCETRACK', self.split)
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} does not exist.")
            return data_info
        if not os.path.exists(depth_dir):
            print(f"Directory {depth_dir} does not exist.")
            return data_info
        if not os.path.exists(app_dir):
            print(f"Directory {app_dir} does not exist.")
            return data_info
        if not os.path.exists(mask_dir):
            print(f"Directory {mask_dir} does not exist.")
            return data_info

        for seq in os.listdir(split_dir):#[8:]:
            seq_path = os.path.join(split_dir, seq)
            depth_path = os.path.join(depth_dir, seq)
            mask_path = os.path.join(mask_dir, seq)
            app_path = os.path.join(app_dir, seq)
            img_path = os.path.join(seq_path, 'img1')
            det_path = os.getenv("detection_path")
            det_path = os.path.join(det_path, *seq_path.split('/')[-3:]) + '.txt'
            # det_path = os.path.join(seq_path, 'gt', 'gt.txt')
            if os.path.isdir(img_path):
                images = sorted([img for img in os.listdir(img_path) if img.endswith('.jpg')])
                depth_feat = sorted([depth for depth in os.listdir(depth_path) if depth.endswith('.pkl')])
                app_feat = sorted([app for app in os.listdir(app_path) if app.endswith('.pkl')])
                mask_feat = sorted([msk for msk in os.listdir(mask_path) if msk.endswith('.pkl')])
                gt_data = self._load_gt(det_path) if os.path.exists(det_path) else {}
                for i in range(len(images)):
                    img1_path = os.path.join(img_path, images[i])
                    frame1 = int(images[i].split('.')[0])
                    det1 = gt_data.get(frame1, [])
                    if not det1:  # Skip if det1 is empty
                        continue
                    matching_path = next(
                        (path for path in depth_feat if int(path.split('.')[0]) == frame1), None)
                    depth1 = os.path.join(depth_path, matching_path)
                    matching_path = next(
                        (path for path in app_feat if int(path.split('.')[0]) == frame1), None)
                    app1 = os.path.join(app_path, matching_path)
                    mask1 = os.path.join(mask_path, matching_path)
                    data_info.append((depth1, app1, mask1, img1_path, np.array(det1)))
            else:
                print(f"Skipping {seq_path} as it does not contain the required files.")
        return data_info

    def _load_gt(self, gt_path):
        gt_data = {}
        with open(gt_path, 'r') as f:
            for line in f:
                frame, id, bb_left, bb_top, bb_width, bb_height, score, *_ = map(float, line.split(','))
                if frame not in gt_data:
                    gt_data[frame] = []
                gt_data[frame].append((1, bb_left, bb_top, bb_width, bb_height, score))  # since we don't have ids
        return gt_data

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        (depth1, app1, mask1, img1_path, det1) = self.data_info[idx]

        img1 = Image.open(img1_path).convert('RGB')

        original_size = img1.size
        if self.transform:
            RGB1 = self.RGB_trf(img1)
            img1 = self.transform(img1)

        # appearance
        with open(app1, 'rb') as f:
            app1 = pickle.load(f)

        app_feat1 = [app1[i][0] for i, bbox1 in enumerate(det1) if np.array_equal(app1[i][1], bbox1)]
        app_feat1 = torch.stack(app_feat1)

        # mask
        with open(mask1, 'rb') as f:
            mask1 = pickle.load(f)

        with open(depth1, 'rb') as f:
            depth1 = pickle.load(f)
        # mask_feat1 =  [(mask1[i][0], mask1[i][1]) for i, bbox1 in enumerate(det1) if np.array_equal(mask1[i][2], bbox1)]
        ms1 = [depth1 * mask1[i][0].int().unsqueeze(0) for
                     i, bbox1 in enumerate(det1) if np.array_equal(mask1[i][2], bbox1)]
        ms2 = [depth1 * mask1[i][1].int().unsqueeze(0) for
                     i, bbox1 in enumerate(det1) if np.array_equal(mask1[i][2], bbox1)]

        ms1 = torch.stack(ms1)
        ms1 = self.depth_transform(ms1)
        ms2 = torch.stack(ms2)
        ms2 = self.depth_transform(ms2)

        mask_feat1 = [(ms1[i], ms2[i]) for i, bbox1 in enumerate(det1) if np.array_equal(mask1[i][2], bbox1)]

        # depth

        depth_feat1 = [depth1 * mask1[i][0].int().unsqueeze(0) for
                     i, bbox1 in enumerate(det1) if np.array_equal(mask1[i][2], bbox1)]

        depth_feat1 = torch.stack(depth_feat1)

        depth_feat1 = self.depth_transform(depth_feat1)

        # # plot depth maps
        # crop1 = depth_feat1[0][0].numpy()
        # plt.imshow(crop1)
        # plt.axis('off')  # Hide axes for better visualization
        # plt.savefig('./plts/test_depth_batch.png')
        # plt.show()

        # depth_feat1 = torch.nn.functional.normalize(depth_feat1.view(depth_feat1.size(0),-1), dim=-1).view(depth_feat1.size(0), 1, 128, 128)

        # depth_feat1 = torch.nn.functional.normalize(depth_feat1.view(depth_feat1.size(0), -1), dim=-1)

        # motion
        motion_feat1 = [det1[i, 1:-1] for i, bbox1 in enumerate(det1)]

        motion_feat1 = np.array(motion_feat1)

        motion_feat1 = torch.from_numpy(motion_feat1).to(dtype=torch.float32)

        # Ensure both lists have the same length (same number of tensors)
        if len(depth_feat1) != len(app_feat1):
            raise ValueError(f"Mismatch in number of tensors: {len(depth_feat1)} vs {len(app_feat1)}")

        det1[:, 0] = None

        return img1, app_feat1, depth_feat1, mask_feat1, motion_feat1, torch.from_numpy(det1),  img1_path, original_size

def collate_fn(batch):
    # Stack image tensors (assuming images are already in tensor format)
    img1_batch = torch.stack([item[0] for item in batch])

    # Stack the feature tensors (assuming features are in tensor format)
    app_feat1_batch = [item[1] for item in batch]
    depth_feat1_batch = [item[2] for item in batch]
    mask_feat1_batch = [item[3] for item in batch]
    motion_feat1_batch = [item[4] for item in batch]

    # Collect the bounding boxes, which may require padding
    bboxes1_batch = [item[5] for item in batch]

    # Paths (just collect them as strings)
    path1_batch = [item[6] for item in batch]

    # Original sizes (just collect them as needed)
    original_size_batch = torch.stack([torch.tensor(item[7]) for item in batch])

    return (
        img1_batch,
        app_feat1_batch,
        depth_feat1_batch,
        mask_feat1_batch,
        motion_feat1_batch,
        bboxes1_batch, path1_batch,
        original_size_batch
    )















