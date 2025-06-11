import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tracking_model import MatchingNetwork
from dataset_eval import DanceTrackDataset, collate_fn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import colorsys
import random
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
import glob
import cv2

class Tracklets(object):
    def __init__(self, img1, bbox1, app1, depth1, mask1, motion1, img1_path):
        self.img = img1
        self.apperance_emb = app1
        self.depth_emb = depth1
        self.mask = mask1
        self.motion_emb = motion1
        self.bbox = bbox1
        age = img1_path.split('/')[-1]
        age = int(age.split('.')[-2])
        self.age = age
        self.img_path = img1_path

    def update_feats(self, bbox1, img1, img1_path, depth1, app1, mask1, motion1, det_score):
        age = img1_path.split('/')[-1]
        age = int(age.split('.')[-2])
        self.bbox = np.squeeze(bbox1) if bbox1.shape[0] >= 1 else bbox1
        self.img = img1
        self.img_path = img1_path
        self.depth_emb = depth1
        trust = (det_score - 0.6) / (1 - 0.6)
        alpha = 0.95
        alpha = alpha + (1 - alpha) * (1 - trust)
        self.apperance_emb = alpha * self.apperance_emb + (1 - alpha) * app1
        self.apperance_emb /= torch.norm(self.apperance_emb, dim=-1, keepdim=True)
        self.apperance_emb = app1
        self.motion_emb = motion1
        self.age = age
        self.mask = mask1

    def get_img(self):
        return self.img

    def get_img_path(self):
        return self.img_path

    def get_depth_emb(self):
        return self.depth_emb

    def get_appearance_emb(self):
        return self.apperance_emb

    def get_mask(self):
        return self.mask

    def get_motion_emb(self):
        return self.motion_emb

    def get_bbox(self):
        return self.bbox

    def get_age(self):
        return self.age



def linear_assignment(cost_matrix):
    matcher = []
    for cost_m in cost_matrix:
        cost = cost_m.detach().cpu().numpy()
        try:
            import lap

            _, x, y = lap.lapjv(cost, extend_cost=True)
            matcher.append(np.array([[y[i], i] for i in x if i >= 0]))  #
        except ImportError:
            from scipy.optimize import linear_sum_assignment

            x, y = linear_sum_assignment(-cost)
            matcher.append(np.array(list(zip(x, y))))
    return matcher

def filter_targets(online_targets, aspect_ratio_thresh=1000, min_box_area=10):
    """
    Removes targets not meeting threshold criteria.

    Args:
        online_targets (np.ndarray): Input array of shape (N, 6), where each row is [id, x1, y1, w, h, score].
        aspect_ratio_thresh (float): Threshold for aspect ratio to consider a target as vertical.
        min_box_area (float): Minimum area for a target to be valid.

    Returns:
        np.ndarray: Filtered array of shape (M, 6), where each row is [id, x1, y1, w, h, score].
    """
    filtered_targets = []
    for t in online_targets:
        tid, x1, y1, w, h, score = t
        # Calculate thresholds
        area = w * h
        aspect_ratio = w / h
        if area > min_box_area and aspect_ratio <= aspect_ratio_thresh:
            filtered_targets.append([tid, x1, y1, w, h, score])
    return np.array(filtered_targets)


# Function to save image pairs with bounding boxes
def visualize_matches(frame_path2, bboxes1, bboxes2, matches, similarity_matrix):
    def convert_bbox_format(bbox):
        """
        Convert (bb_left, bb_top, bb_width, bb_height) to (x1, y1, x2, y2).
        """
        bb_left, bb_top, bb_width, bb_height = (bbox)
        x1 = int(bb_left)
        y1 = int(bb_top)
        x2 = int(bb_left + bb_width)
        y2 = int(bb_top + bb_height)
        return (x1, y1, x2, y2)
    """
    Visualize matched bounding boxes between two frames.
    """
    # Extract directory and filename
    dir_path, filename = os.path.split(frame_path2)

    # Extract the number and decrement
    num, ext = os.path.splitext(filename)  # "00000002", ".jpg"
    prev_num = int(num) - 1  # Convert to int and subtract 1
    prev_filename = f"{prev_num:08d}{ext}"  # Format back to 8-digit format

    # Get the previous image path
    frame_path1 = os.path.join(dir_path, prev_filename)

    frame1 = cv2.imread(frame_path1)  # Replace with the first frame
    frame2 = cv2.imread(frame_path2)  # Replace with the second frame
    combined = np.hstack((frame1, frame2))
    H1, W1 = frame1.shape[:2]

    plt.figure(figsize=(15, 10))
    plt.imshow(combined)
    plt.axis("off")

    for i, j in matches:
        # Generate a random color for each matched pair
        color = np.random.rand(3, )
        x1, y1, x2, y2 = convert_bbox_format(bboxes1[i][1:-1])
        x1_next, y1_next, x2_next, y2_next = convert_bbox_format(bboxes2[j][1:])
        x1_next += W1
        x2_next += W1

        # Draw bounding boxes with the same color
        plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, linewidth=2, fill=False))
        plt.gca().add_patch(
            Rectangle((x1_next, y1_next), x2_next - x1_next, y2_next - y1_next, edgecolor=color, linewidth=2,
                      fill=False))

        # Draw lines connecting matched boxes
        plt.plot([x1 + (x2 - x1) // 2, x1_next + (x2_next - x1_next) // 2],
                 [y1 + (y2 - y1) // 2, y1_next + (y2_next - y1_next) // 2], color="cyan", linewidth=0.5)

        # Add similarity score as label
        plt.text(x1, y1 - 10, f"Score: {similarity_matrix[i, j]:.2f}", color="white", fontsize=8,
                 bbox=dict(facecolor='red', alpha=0.5))

    plt.title("Bounding Box Matches")
    plt.savefig('./test_attention_embedded_feat/plts/{batch_idx}.png')
    # plt.show()
    plt.close()

def save_image_pairs_batch1(img1, img2, bboxes1, bboxes2, matches, batch_idx, original_sizes):
    batch_size = img1.size(0)
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, batch_size * 5))

    for i in range(batch_size):
        img1_np = img1[i].permute(1, 2, 0).cpu().numpy()
        img2_np = img2[i].permute(1, 2, 0).cpu().numpy()

        orig_w, orig_h = original_sizes[i]
        _, img_h, img_w = img1[i].size()
        scale_x = img_w / orig_w
        scale_y = img_h / orig_h

        axes[i, 0].imshow(img1_np)
        axes[i, 1].imshow(img2_np)

        used_colors = {}

        # Draw all bounding boxes with ground truth matches
        for (id1, x1, y1, w1, h1) in bboxes1[i]:
            if id1 in matches[i]:
                color = get_distinguishable_color()
                used_colors[id1] = color
                rect = Rectangle((x1 * scale_x, y1 * scale_y), w1 * scale_x, h1 * scale_y, edgecolor=color,
                                 facecolor='none', linewidth=2)
            else:
                rect = Rectangle((x1 * scale_x, y1 * scale_y), w1 * scale_x, h1 * scale_y, edgecolor='red',
                                 facecolor='none', linewidth=3)

            axes[i, 0].text(x1 * scale_x, y1 * scale_y, str(id1), color='white', fontsize=5, verticalalignment='top')
            axes[i, 0].add_patch(rect)

        for (id2, x2, y2, w2, h2) in bboxes2[i]:
            #match_id = next((id1 for id1, id2_ in matches[i].items() if id2_ == id2), None)
            match_id = next((id1 for id1, id2_ in matches[i].items() if id2_ == id2), None)
            if match_id is not None:
                color = used_colors[match_id]
                rect = Rectangle((x2 * scale_x, y2 * scale_y), w2 * scale_x, h2 * scale_y, edgecolor=color,
                                 facecolor='none', linewidth=2)
            else:
                rect = Rectangle((x2 * scale_x, y2 * scale_y), w2 * scale_x, h2 * scale_y, edgecolor='red',
                                 facecolor='none', linewidth=3)

            axes[i, 1].text(x2 * scale_x, y2 * scale_y, str(id2), color='white', fontsize=6, verticalalignment='top')
            axes[i, 1].add_patch(rect)

        axes[i, 0].axis('off')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f'./test_image_pairs_batch_{batch_idx}.png')
    plt.close(fig)


def save_image_pairs(img1, img2, bboxes1, bboxes2, matches, batch_idx, original_sizes):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    img1_np = img1[0].permute(1, 2, 0).cpu().numpy()
    img2_np = img2[0].permute(1, 2, 0).cpu().numpy()

    orig_w, orig_h = original_sizes[0]
    _, img_h, img_w = img1[0].size()
    scale_x = img_w / orig_w
    scale_y = img_h / orig_h

    axes[0].imshow(img1_np)
    axes[1].imshow(img2_np)

    used_colors = {}

    # Draw all bounding boxes with ground truth matches
    for (id1, x1, y1, w1, h1), _, _ in bboxes1[0]:
        if id1 in matches[0]:
            color = get_distinguishable_color()
            used_colors[id1] = color
            rect = Rectangle((x1 * scale_x, y1 * scale_y), w1 * scale_x, h1 * scale_y, edgecolor=color,
                             facecolor='none', linewidth=2)
        else:
            rect = Rectangle((x1 * scale_x, y1 * scale_y), w1 * scale_x, h1 * scale_y, edgecolor='red',
                             facecolor='none', linewidth=3)

        axes[0].text(x1 * scale_x, y1 * scale_y, str(id1), color='white', fontsize=5, verticalalignment='top')
        axes[0].add_patch(rect)

    for (id2, x2, y2, w2, h2), _, _ in bboxes2[0]:
        match_id = next((id1 for id1, id2_ in matches[0].items() if id2_ == id2), None)
        if match_id is not None:
            color = used_colors[match_id]
            rect = Rectangle((x2 * scale_x, y2 * scale_y), w2 * scale_x, h2 * scale_y, edgecolor=color,
                             facecolor='none', linewidth=2)
        else:
            rect = Rectangle((x2 * scale_x, y2 * scale_y), w2 * scale_x, h2 * scale_y, edgecolor='red',
                             facecolor='none', linewidth=3)

        axes[1].text(x2 * scale_x, y2 * scale_y, str(id2), color='white', fontsize=6, verticalalignment='top')
        axes[1].add_patch(rect)

    axes[0].axis('off')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'./test_image_pairs_batch_{batch_idx}.png')
    plt.close(fig)


# Generate random colors for matching bounding boxes
def get_random_color():
    return [random.random() for _ in range(3)]


# Global list to store previously chosen colors
previous_colors = []


def get_distinguishable_color():
    global previous_colors

    # Define a list of predefined hues with random variations
    predefined_hues = [0.0, 1 / 6, 1 / 3, 0.5, 2 / 3, 5 / 6, 0.1, 0.9, 0.4, 0.8, 0.2, 0.7, 0.25, 0.55, 0.35, 0.85, 0.15,
                       0.6, 0.45, 0.3, 0.65]

    # Select a random predefined hue and ensure it's within [0, 1]
    hue = random.choice(predefined_hues) % 1.0

    # Generate random saturation and value to add some variability
    saturation = 0.7 + random.random() * 0.3  # Ensuring saturation is not too low
    value = 0.7 + random.random() * 0.3  # Ensuring value is not too low

    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)

    # Convert RGB tuple to a hashable format (tuple of floats)
    color_key = tuple(rgb)

    # Check if the color has been used before; if so, choose a different one
    while predefined_hues in previous_colors:
        predefined_hues = [
            h / 20 + random.uniform(-0.05, 0.05)
            for h in range(20)
        ]
        hue = random.choice(predefined_hues) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color_key = tuple(rgb)

    # Add the new color to the list of previous colors
    previous_colors.append(predefined_hues)

    # Limit the list to a reasonable length to avoid excessive memory usage
    if len(previous_colors) > 100:
        previous_colors = previous_colors[-100:]

    return rgb

def sort_and_save_tracked_objects(tracked_objs, output_dir='./test_attention_embedded_feat/results/DANCE-val/DANCETRACK/data'):
    """
    Sort the tracked objects by ID for each key and save them to .txt files.

    Args:
    - tracked_objs (dict): A dictionary where the key is a sequence name (str)
                           and the value is a NumPy array with columns [id, x, y, w, h, score].
    - output_dir (str): Directory to save the sorted files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for sequence_name, data in tracked_objs.items():
        # Use np.lexsort for multi-key sorting: sort by frame (last column) first, then by id (first column)
        sort_indices = np.lexsort((data[:, 0], data[:, 6]))  # Sort by id (col 0), then by frame (col 6)
        sorted_data = data[sort_indices]

        # Save the sorted data to a file
        output_file = os.path.join(output_dir, f"{sequence_name}.txt")
        with open(output_file, 'w') as f:
            for row in sorted_data:
                id, x, y, w, h, score, frame = row
                f.write(f"{int(frame)},{int(id)},{x},{y},{w},{h},-1,-1,-1,-1\n")

    print(f"Tracked objects of sequence {sequence_name}.")

def dti(txt_path, save_path, n_min=30, n_dti=20):
    def dti_write_results(filename, results):
        # os.makedirs(filename, exist_ok=True)
        save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
        with open(filename, "w") as f:
            for i in range(results.shape[0]):
                frame_data = results[i]
                frame_id = int(frame_data[0])
                track_id = int(frame_data[1])
                x1, y1, w, h = frame_data[2:6]
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
                f.write(line)

    seq_txts = sorted(glob.glob(os.path.join(txt_path, "*.txt")))
    # breakpoint()
    for seq_txt in seq_txts:
        seq_name = seq_txt.replace("\\", "/").split("/")[-1]  ## To better play along with windows paths
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=",")
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)
        for track_id in range(min_id, max_id + 1):
            index = seq_data[:, 1] == track_id
            tracklet = seq_data[index]
            tracklet_dti = tracklet
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]
            n_conf = np.sum(tracklet[:, 6] > 0.5)
            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / (
                                right_frame - left_frame
                            ) + left_bbox
                            frames_dti[curr_frame] = curr_bbox
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))
            seq_results = np.vstack((seq_results, tracklet_dti))
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        dti_write_results(save_seq_txt, seq_results)







