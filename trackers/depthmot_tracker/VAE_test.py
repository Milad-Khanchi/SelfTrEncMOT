import sys

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from torchvision import transforms
import torch.nn.functional as F

transform_depth = transforms.Compose([
    transforms.Lambda(lambda x: torch.nn.functional.interpolate(
        x, size=(128, 128), mode='bicubic', align_corners=False)),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    # transforms.Lambda(lambda x: x.half())  # Convert to float16
])


class MatchingNetwork(nn.Module):
    def __init__(self, feature_dim_motion=4, feature_dim_depth=4096, feature_dim_appr=2048):
        super(MatchingNetwork, self).__init__()
        self.auto_encoder_depth = ConvAutoencoder()

    def forward(self, mask1, mask2):
        depth_emb1 = self.auto_encoder_depth(mask1)

        depth_emb2 = self.auto_encoder_depth(mask2)

        emb1 = F.normalize(depth_emb1, p=2, dim=-1)
        emb2 = F.normalize(depth_emb2, p=2, dim=-1)

        return emb1, emb2


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(  
            nn.Conv2d(1, 32, kernel_size=(4, 4), stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2, padding=1),  # (batch_size, 64, 96, 96)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1),  # (batch_size, 128, 48, 48)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output
            nn.Linear(128 * 16 * 16, 2048)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2048, 128 * 16 * 16),  # Linear layer to match encoder output
            nn.Unflatten(1, (128, 16, 16)),  # Unflatten back to (batch_size, 128, 31, 18)
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1),  # (batch_size, 64, 62, 34)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=2, padding=1),  # (batch_size, 32, 63, 33)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(4, 4), stride=2, padding=1),  # (batch_size, 1, 64, 32)
        )


    def forward(self, x):
        head1_output = self.encoder(x)
        # decoded = self.decoder(head1_output)

        # # plot
        # # Move to CPU & detach for plotting
        # input_images = x.cpu().detach().numpy()
        # reconstructed_images = decoded.cpu().detach().numpy()
        #
        # # Number of images to plot (limit batch size if needed)
        # num_images = min(5, x.shape[0])
        #
        # # Create a figure with subplots (2 rows: inputs & outputs)
        # fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))
        #
        # for i in range(num_images):
        #     # Plot input images (grayscale)
        #     axes[0, i].imshow(input_images[i, 0], cmap='gray')
        #     axes[0, i].set_title("Input")
        #     axes[0, i].axis("off")
        #
        #     # Plot reconstructed images (grayscale)
        #     axes[1, i].imshow(reconstructed_images[i, 0], cmap='gray')
        #     axes[1, i].set_title("Reconstructed")
        #     axes[1, i].axis("off")
        #
        # plt.tight_layout()
        #
        # # Save the figure (avoid blocking execution)
        # plt.savefig("/cache/plts/autoencoder_results.png")
        # plt.close(fig)  # Close figure to free memory

        return head1_output









