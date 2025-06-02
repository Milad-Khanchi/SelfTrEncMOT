import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



class DanceTrackDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
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
            if os.path.isdir(img_path):
                images = sorted([img for img in os.listdir(img_path) if img.endswith('.jpg')])
                for i in range(len(images)):
                    img1_path = os.path.join(img_path, images[i])
                    frame1 = int(images[i].split('.')[0])
                    data_info.append((frame1, img1_path))
            else:
                print(f"Skipping {seq_path} as it does not contain the required files.")
        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        (frame1, img1_path) = self.data_info[idx]

        img1 = Image.open(img1_path).convert('RGB')
        original_size = img1.size
        if self.transform:
            img1 = self.transform(img1)


        return img1.unsqueeze(0), frame1, img1_path, original_size


def collate_fn(batch):
    img1_batch = batch[0][0]
    frame1_batch = batch[0][1]
    path1_batch = batch[0][2]
    original_size_batch = batch[0][3]

    return img1_batch, frame1_batch, path1_batch, original_size_batch












