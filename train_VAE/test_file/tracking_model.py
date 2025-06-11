import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt

class MatchingNetwork(nn.Module):
    def __init__(self, feature_dim_motion=4, feature_dim_depth=4096, feature_dim_appr=2048):
        super(MatchingNetwork, self).__init__()
        self.auto_encoder_depth = ConvAutoencoder()

    def forward(self, app1, app2, depth1, depth2, mask1, mask2, motion1, motion2, device):
       
        # _, mask_h, mask_w = mask1[0].shape
        # mask1 = [mask1[0].view(mask1[0].size(0), -1)]
        # mask2 = [mask2[0].view(mask2[0].size(0), -1)]
        #
        # padded_keys_dict, key_padding_masks_dict = self.feat_padding(app1, app2, depth1, depth2, mask1, mask2, motion1, motion2, device)

        # depth1_auto_in = padded_keys_dict['depth_feat1'].view(padded_keys_dict['depth_feat1'].size(0),
        #                                                       padded_keys_dict['depth_feat1'].size(1), 1, 224, 224)
        # depth2_auto_in = padded_keys_dict['depth_feat2'].view(padded_keys_dict['depth_feat2'].size(0),
        #                                                       padded_keys_dict['depth_feat1'].size(1), 1, 224, 224)
        #
        # padded_keys_dict['mask1'] = padded_keys_dict['mask1'].view(padded_keys_dict['mask1'].size(0),
        #                                      padded_keys_dict['mask1'].size(1), mask_h, mask_w)
        #
        # padded_keys_dict['mask2'] = padded_keys_dict['mask2'].view(padded_keys_dict['mask2'].size(0),
        #                                                            padded_keys_dict['mask2'].size(1), mask_h, mask_w)

        depth_emb1, _ = self.auto_encoder_depth(mask1[0].to(device))

        depth_emb2, _ = self.auto_encoder_depth(mask2[0].to(device))

        emb1 = F.normalize(depth_emb1, p=2, dim=-1)
        emb2 = F.normalize(depth_emb2, p=2, dim=-1)
        depth_score = torch.bmm(emb1.unsqueeze(0), emb2.unsqueeze(0).transpose(-2, -1))

        motion_score = self.motion_sim(motion1[0], motion2[0])
        #
        # mask_score = self.mask_iou_batch(mask1[0], mask2[0])
        #
        # motion_score *= torch.exp(mask_score)

        # # high motion similarity and high mask IoU
        # row_max_condition = motion_score == motion_score.max(dim=1, keepdim=True).values
        # col_max_condition = motion_score == motion_score.max(dim=0, keepdim=True).values
        # condition = (motion_score > 0.1) & row_max_condition & col_max_condition
        #
        # # Assign a large value to emphasize matches meeting the condition
        # boost_value = 10  # Adjust this value as needed
        # boost_matrix = condition.float() * boost_value
        #
        # app_score = torch.bmm(app1[0].unsqueeze(0), app2[0].unsqueeze(0).transpose(-2, -1))

        final_score = depth_score #(motion_score).unsqueeze(0) # + app_score + boost_matrix.unsqueeze(0)

        return motion_score.unsqueeze(0), final_score, final_score

    def feat_padding(self, app_feat1, app_feat2, depth_feat1, depth_feat2, mask1, mask2, motion_feat1, motion_feat2, device):
        # Dictionary to store padded tensors and masks
        padded_keys_dict = {}
        key_padding_masks_dict = {}

        # Determine the maximum sequence length in the batch
        max_seq_len = 0
        for key in app_feat1 + app_feat2:
            max_seq_len = max(max_seq_len, key.size(0))

        # Loop through each list to pad tensors and create masks
        for key_name, key_list in zip(
                ['app_feat1', 'app_feat2', 'depth_feat1', 'depth_feat2', 'mask1', 'mask2', 'motion_feat1', 'motion_feat2'],
                [app_feat1, app_feat2, depth_feat1, depth_feat2, mask1, mask2, motion_feat1, motion_feat2]):

            # Pad tensors to the same sequence length
            padded_tensors = [torch.nn.functional.pad(t, (0, 0, 0, max_seq_len - t.size(0))) for t in key_list]

            # Stack tensors to create a batch
            padded_keys = torch.stack(padded_tensors)  # Shape: (batch_size, max_seq_len, 2048)

            # Create a mask where padded positions are set to True
            key_padding_mask = torch.zeros((padded_keys.size(0), max_seq_len), dtype=torch.bool)

            # Populate the mask
            for i, t in enumerate(key_list):
                key_padding_mask[i, t.size(0):] = True  # Mask out padding positions

            # Store padded tensors and masks in corresponding variables
            padded_keys_dict[key_name] = padded_keys.to(device)
            key_padding_masks_dict[key_name] = key_padding_mask.to(device)

        return padded_keys_dict, key_padding_masks_dict

    def motion_sim(self, bboxes1, bboxes2):
        """
            Computes IoU between two sets of bounding boxes in the format [x1, y1, w, h].
            Args:
                bboxes1: Tensor of shape (N, 4).
                bboxes2: Tensor of shape (M, 4).
            Returns:
                Tensor of shape (N, M) containing the IoU values.
            """
        # Convert [x1, y1, w, h] to [x1, y1, x2, y2]
        bboxes1 = torch.cat((bboxes1[:, :2], bboxes1[:, :2] + bboxes1[:, 2:]), dim=1)
        bboxes2 = torch.cat((bboxes2[:, :2], bboxes2[:, :2] + bboxes2[:, 2:]), dim=1)

        # Add dimensions to allow broadcasting
        bboxes2 = bboxes2.unsqueeze(0)  # Shape: (1, M, 4)
        bboxes1 = bboxes1.unsqueeze(1)  # Shape: (N, 1, 4)

        # Calculate intersection coordinates
        xx1 = torch.max(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = torch.max(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = torch.min(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = torch.min(bboxes1[..., 3], bboxes2[..., 3])

        # Calculate intersection width and height
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        intersection = w * h

        # Calculate area of both bounding boxes
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

        # Calculate union area
        union = area1 + area2 - intersection

        # Compute IoU
        iou = intersection / union
        return iou

    def mask_iou_batch(self, masks1, masks2):
        """
        Compute IoU between two sets of segmentation masks.

        :param masks1: First set of binary masks (shape: N x H x W)
        :param masks2: Second set of binary masks (shape: M x H x W)
        :return: IoU scores for each mask pair (shape: N x M)
        """
        # Ensure masks are boolean tensors
        masks1 = masks1.bool()  # Convert to boolean if not already
        masks2 = masks2.bool()

        # Expand masks to broadcast the dimensions
        masks1 = masks1.unsqueeze(1)  # Shape: (N, 1, H, W)
        masks2 = masks2.unsqueeze(0)  # Shape: (1, M, H, W)

        # Compute intersection and union
        intersection = torch.logical_and(masks1, masks2).sum(dim=(2, 3))  # Intersection: N x M
        union = torch.logical_or(masks1, masks2).sum(dim=(2, 3))  # Union: N x M

        # Compute IoU, avoiding division by zero
        iou = intersection / union
        iou[union == 0] = 0  # Avoid division by zero (IoU is 0 where union is 0)

        return iou




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
        decoded = self.decoder(head1_output)
        
        # plot
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
        #     axes[0, i].imshow(input_images[i, 0])
        #     axes[0, i].set_title("Input")
        #     axes[0, i].axis("off")
        # 
        #     # Plot reconstructed images (grayscale)
        #     axes[1, i].imshow(reconstructed_images[i, 0])
        #     axes[1, i].set_title("Reconstructed")
        #     axes[1, i].axis("off")
        # 
        # plt.tight_layout()
        # 
        # # Save the figure (avoid blocking execution)
        # plt.savefig("./plts/autoencoder_results.png")
        # plt.close(fig)  # Close figure to free memory
        
        return head1_output, decoded  # head1_output.view(batch_size, seq_size, 1536)#, decoded.view(batch_size, seq_size, -1)


