import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tracking_model import MatchingNetwork
from dataset import DanceTrackDataset, collate_fn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import colorsys
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR




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


# # Global list to store previously chosen colors
# global previous_colors
# previous_colors = []


def get_distinguishable_color():
    global previous_colors
    previous_colors = []

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


# Function to save image pairs with bounding boxes
def save_image_pairs(img1, img2, bboxes1, bboxes2, matches, epoch, batch_idx, original_sizes, writer, mode):
    #batch_size = img1.size(0)
    batch_size = 4
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
        for box1 in bboxes1[i]:
            (id1, x1, y1, w1, h1, _) = box1.tolist()
            if id1 in matches[i]:
                color = get_distinguishable_color()
                used_colors[id1] = color
                rect = Rectangle((x1 * scale_x, y1 * scale_y), w1 * scale_x, h1 * scale_y, edgecolor=color,
                                 facecolor='none', linewidth=2)
                axes[i, 0].text(x1 * scale_x, y1 * scale_y, str(id1), color='white', fontsize=6,
                                verticalalignment='top')
            else:
                rect = Rectangle((x1 * scale_x, y1 * scale_y), w1 * scale_x, h1 * scale_y, edgecolor='red',
                                 facecolor='none', linewidth=3)
            axes[i, 0].add_patch(rect)

        for box2 in bboxes2[i]:
            (id2, x2, y2, w2, h2, _) = box2.tolist()
            match_id = next((id1 for id1, id2_ in matches[i].items() if id2_ == id2), None)
            if match_id is not None:
                color = used_colors[match_id]
                rect = Rectangle((x2 * scale_x, y2 * scale_y), w2 * scale_x, h2 * scale_y, edgecolor=color,
                                 facecolor='none', linewidth=2)
                axes[i, 1].text(x2 * scale_x, y2 * scale_y, str(id2), color='white', fontsize=6,
                                verticalalignment='top')
            else:
                rect = Rectangle((x2 * scale_x, y2 * scale_y), w2 * scale_x, h2 * scale_y, edgecolor='red',
                                 facecolor='none', linewidth=3)
            axes[i, 1].add_patch(rect)

        axes[i, 0].axis('off')
        axes[i, 1].axis('off')

    #plt.tight_layout()
    if mode == 'train':
        batch_idx = 'train'
    else:
        batch_idx = 'test'
    # writer.add_figure(f'Image Pairs Epoch {epoch} Batch {batch_idx} {mode}', fig)
    plt.savefig(f'./train_attention_embedded_feat/plts/_{batch_idx}.png')
    plt.close(fig)
    global previous_colors
    previous_colors = []