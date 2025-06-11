import os
from torch.utils.data import Dataset

from dotenv import load_dotenv
load_dotenv()

class DanceTrackDataset_test(Dataset):
    def __init__(self, root_dir, split='test', mode='train'):
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.data_info = self._load_data_info()

    def _load_data_info(self):
        data_info = []
        split_dir = os.path.join(self.root_dir, 'DANCETRACK', self.split)
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} does not exist.")
            return data_info

        for seq in os.listdir(split_dir):
            seq_path = os.path.join(split_dir, seq)
            img_path = os.path.join(seq_path, 'img1')
            if self.mode == 'test':
                det_path = os.getenv("detection_path")
                det_path = os.path.join(det_path, *seq_path.split('/')[-3:]) + '.txt'
            else:
                det_path = os.path.join(seq_path, 'gt', 'gt.txt')
            if os.path.isdir(img_path):
                images = sorted([img for img in os.listdir(img_path) if img.endswith('.jpg')])
                det_path = self._load_det(det_path) if os.path.exists(det_path) else {}
                for i in range(len(images)):
                    img1_path = os.path.join(img_path, images[i])
                    frame1 = int(images[i].split('.')[0])
                    det1 = det_path.get(frame1, [])
                    if not det1:  # Skip if det1 is empty
                        continue
                    data_info.append((img1_path, det1))
            else:
                print(f"Skipping {seq_path} as it does not contain the required files.")
        return data_info

    def _load_det(self, det_path):
        det_data = {}
        with open(det_path, 'r') as f:
            for counter, line in enumerate(f):
                frame, id, bb_left, bb_top, bb_width, bb_height, score, *_ = map(float, line.split(','))
                if frame not in det_data:
                    det_data[frame] = []
                if self.split == 'test':
                    det_data[frame].append((None, bb_left, bb_top, bb_width, bb_height, score))  # since we don't have ids
                else:
                    det_data[frame].append((id, bb_left, bb_top, bb_width, bb_height, score))
        return det_data

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        seq_path, dets = self.data_info[idx]
        return seq_path, dets

def collate_fn(batch):
    return batch[0]

