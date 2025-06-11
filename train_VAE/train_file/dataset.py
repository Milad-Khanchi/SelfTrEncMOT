import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
from torchvision import transforms
import numpy as np
import timm
import random

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
    def __init__(self, root_dir, depth_feat_dir, app_feat_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.depth_feat_dir = depth_feat_dir
        self.app_feat_dir = app_feat_dir
        self.split = split
        self.transform = transform
        self.depth_transform = transform_depth
        self.data_info = self._load_data_info()
        self.RGB_trf = RGB_trs

    def _load_data_info(self):
        data_info = []
        split_dir = os.path.join(self.root_dir, 'DANCETRACK', self.split)
        depth_dir = os.path.join(self.depth_feat_dir, 'DANCETRACK', self.split)
        app_dir = os.path.join(self.app_feat_dir, 'DANCETRACK', self.split)
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} does not exist.")
            return data_info
        if not os.path.exists(depth_dir):
            print(f"Directory {depth_dir} does not exist.")
            return data_info
        if not os.path.exists(app_dir):
            print(f"Directory {app_dir} does not exist.")
            return data_info

        for seq in os.listdir(split_dir):#[:5]
            seq_path = os.path.join(split_dir, seq)
            depth_path = os.path.join(depth_dir, seq)
            app_path = os.path.join(app_dir, seq)
            img_path = os.path.join(seq_path, 'img1')
            gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
            if os.path.isdir(img_path):
                images = sorted([img for img in os.listdir(img_path) if img.endswith('.jpg')])
                depth_feat = sorted([depth for depth in os.listdir(depth_path) if depth.endswith('.pkl')])
                app_feat = sorted([app for app in os.listdir(app_path) if app.endswith('.pkl')])
                gt_data = self._load_gt(gt_path) if os.path.exists(gt_path) else {}
                for i in range(len(images)): #range(10):
                    img1_path = os.path.join(img_path, images[i])
                    frame1 = int(images[i].split('.')[0])
                    gt1 = gt_data.get(frame1, [])
                    depth1 = os.path.join(depth_path, depth_feat[i])
                    app1 = os.path.join(app_path, app_feat[i])
                    data_info.append((depth1, app1, img1_path, np.array(gt1)))
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
                if self.split == 'test':
                    gt_data[frame].append((None, bb_left, bb_top, bb_width, bb_height, score))  # since we don't have ids
                else:
                    gt_data[frame].append((id, bb_left, bb_top, bb_width, bb_height, score))
        return gt_data

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        (depth1, app1, img1_path, gt1) =  self.data_info[idx]

        end = min(len(self.data_info) - 1, idx + 1)  # +6 to include idx+5

        # # Randomly choose an index within the range
        # random_idx = random.randint(idx, end - 1)

        (depth2, app2, img2_path, gt2) = self.data_info[end]

        if img1_path.split('/')[-3] != img2_path.split('/')[-3]:
            (depth2, app2, img2_path, gt2) = (depth1, app1, img1_path, gt1)

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        original_size = img1.size
        if self.transform:
            RGB1 = self.RGB_trf(img1)
            RGB2 = self.RGB_trf(img2)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        #appearance
        with open(app1, 'rb') as f:
            app1 = pickle.load(f)

        with open(app2, 'rb') as f:
            app2 = pickle.load(f)

        # depth
        with open(depth1, 'rb') as f:
            depth1 = pickle.load(f)

        with open(depth2, 'rb') as f:
            depth2 = pickle.load(f)

        # app_feat1 = [torch.cat((RGB1 * app1[i][0].int().unsqueeze(0).repeat(3,1,1), depth1 * app1[i][0].int().unsqueeze(0), app1[i][0].int().unsqueeze(0)), dim=0) for i, bbox1 in enumerate(gt1) if np.array_equal(app1[i][2], bbox1)]
        # app_feat2 = [torch.cat((RGB2 * app2[j][0].int().unsqueeze(0).repeat(3,1,1), depth2 * app2[j][0].int().unsqueeze(0), app2[j][0].int().unsqueeze(0)), dim=0) for j, bbox2 in enumerate(gt2) if np.array_equal(app2[j][2], bbox2)]

        app_feat1 = [depth1 * app1[i][0].int().unsqueeze(0) for
                     i, bbox1 in enumerate(gt1) if np.array_equal(app1[i][2], bbox1)]
        app_feat2 = [depth1 * app2[j][1].int().unsqueeze(0) for
                     j, bbox2 in enumerate(gt2) if np.array_equal(app2[j][2], bbox2)]

        app_feat1 = torch.stack(app_feat1)
        app_feat2 = torch.stack(app_feat2)

        app_feat1_f = self.depth_transform(app_feat1).view(app_feat1.size(0), -1)
        app_feat2_f = self.depth_transform(app_feat2).view(app_feat2.size(0), -1)

        # # plot seg maps
        # crop1 = app_feat1_f[0][:3].permute(1, 2, 0).numpy()
        # plt.imshow(crop1)
        # plt.axis('off')  # Hide axes for better visualization
        # plt.savefig('./plts/test_seg_batch.png')
        # plt.show()

        # # plot depth maps
        # crop1 = app_feat1_f[0][0].numpy()
        # plt.imshow(crop1)
        # plt.axis('off')  # Hide axes for better visualization
        # plt.savefig('./plts/test_depth_batch.png')
        # plt.show()

        # motion
        motion_feat1 = [gt1[i, 1:-1] for i, bbox1 in enumerate(gt1) if np.array_equal(app1[i][2], bbox1)]
        motion_feat2 = [gt2[j, 1:-1] for j, bbox2 in enumerate(gt2) if np.array_equal(app2[j][2], bbox2)]

        motion_feat1 = np.array(motion_feat1)
        motion_feat2 = np.array(motion_feat2)

        motion_feat1 = torch.nn.functional.normalize(torch.from_numpy(motion_feat1).to(dtype=torch.float32), dim=-1)
        motion_feat2 = torch.nn.functional.normalize(torch.from_numpy(motion_feat2).to(dtype=torch.float32), dim=-1)

        # Shuffle gt_matches and other features consistently
        num_gt1 = len(gt1)
        num_gt2 = len(gt2)

        # Create a random permutation of indices
        perm1 = torch.randperm(num_gt1)
        perm2 = torch.randperm(num_gt2)

        # Shuffle
        gt1 = torch.tensor(gt1)[perm1]
        gt2 = torch.tensor(gt2)[perm2]

        app_feat1_f = app_feat1_f[perm1]
        app_feat2_f = app_feat2_f[perm2]
        app_feat1 = app_feat1[perm1]
        app_feat2 = app_feat2[perm2]
        motion_feat1 = motion_feat1[perm1]
        motion_feat2 = motion_feat2[perm2]

        # # Ensure both lists have the same length (same number of tensors)
        # if len(depth_feat1) != len(app_feat1):
        #     raise ValueError(f"Mismatch in number of tensors: {len(depth_feat1)} vs {len(app_feat1)}")
        #
        # # Ensure both lists have the same length (same number of tensors)
        # if len(depth_feat2) != len(app_feat2):
        #     raise ValueError(f"Mismatch in number of tensors: {len(depth_feat2)} vs {len(app_feat2)}")

        # Create a ground truth matching matrix (binary)
        gt_matches = torch.zeros((len(gt1), len(gt2)), dtype=torch.float32)
        for i, (id1, *_) in enumerate(gt1):
            for j, (id2, *_) in enumerate(gt2):
                if id1 == id2:
                    gt_matches[i, j] = id1 + 1

        return img1, img2, app_feat1_f, app_feat2_f, motion_feat1, motion_feat2, gt1, gt2, gt_matches,  img1_path, img2_path, original_size

def collate_fn(batch):
    # Stack image tensors (assuming images are already in tensor format)
    img1_batch = torch.stack([item[0] for item in batch])
    img2_batch = torch.stack([item[1] for item in batch])

    # Stack the feature tensors (assuming features are in tensor format)
    app_feat1_batch = [item[2] for item in batch]
    app_feat2_batch = [item[3] for item in batch]
    motion_feat1_batch = [item[4] for item in batch]
    motion_feat2_batch = [item[5] for item in batch]

    # Collect the bounding boxes, which may require padding
    bboxes1_batch = [item[6] for item in batch]
    bboxes2_batch = [item[7] for item in batch]

    # Ground truth matches, assuming they may vary in size (need padding)
    gt_matches_batch = [item[8] for item in batch]

    # Paths (just collect them as strings)
    path1_batch = [item[9] for item in batch]
    path2_batch = [item[10] for item in batch]

    # Original sizes (just collect them as needed)
    original_size_batch = torch.stack([torch.tensor(item[11]) for item in batch])

    # Find the maximum number of bounding boxes in the batch
    max_bboxes1 = max(len(b) for b in bboxes1_batch)
    max_bboxes2 = max(len(b) for b in bboxes2_batch)
    max_rows = max(max_bboxes1, max_bboxes2)

    # Pad the ground truth matches if necessary
    padded_gt_matches_batch = torch.zeros((len(gt_matches_batch), max_rows, max_rows), dtype=torch.float32)
    for i, gt_matches in enumerate(gt_matches_batch):
        padded_gt_matches_batch[i, :gt_matches.size(0), :gt_matches.size(1)] = gt_matches

    return (
        img1_batch, img2_batch,
        app_feat1_batch, app_feat2_batch,
        motion_feat1_batch, motion_feat2_batch,
        bboxes1_batch, bboxes2_batch,
        padded_gt_matches_batch, path1_batch, path2_batch, original_size_batch
    )















