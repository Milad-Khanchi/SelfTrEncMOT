import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt

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
        return  head1_output, decoded #head1_output.view(batch_size, seq_size, 1536)#, decoded.view(batch_size, seq_size, -1)


class MatchingNetwork(nn.Module):
    def __init__(self, feature_dim_motion=4, feature_dim_depth=4096, feature_dim_appr=2048):
        super(MatchingNetwork, self).__init__()
        self.auto_encoder_depth = ConvAutoencoder()

    def forward(self, app1, app2, motion1, motion2, device):
        
        #padded_keys_dict, key_padding_masks_dict = self.feat_padding(app1, app2, motion1, motion2, device)

        # depth1_auto_in = padded_keys_dict['app_feat1'].view(padded_keys_dict['app_feat1'].size(0), padded_keys_dict['app_feat1'].size(1), 5, 224, 224)
        # depth2_auto_in = padded_keys_dict['app_feat2'].view(padded_keys_dict['app_feat2'].size(0), padded_keys_dict['app_feat2'].size(1), 5, 224, 224)

        # out1 = (depth1_auto_in[:,:,-1,:,:] > 0.5).int().view(depth1_auto_in.size(0), depth1_auto_in.size(1), -1)
        #
        # out2 = (depth2_auto_in[:,:,-1,:,:] > 0.5).int().view(depth2_auto_in.size(0), depth2_auto_in.size(1), -1)

        # # plot seg maps
        # crop1 = out1[0,0,:,:].cpu().numpy()
        # plt.imshow(crop1)
        # plt.axis('off')  # Hide axes for better visualization
        # plt.savefig('./train_attention_embedded_feat/plts/test_seg_batch.png')
        # plt.show()
        #
        # # plot seg maps
        # crop1 = out2[0, 0, :, :].cpu().numpy()
        # plt.imshow(crop1)
        # plt.axis('off')  # Hide axes for better visualization
        # plt.savefig('./train_attention_embedded_feat/plts/test_seg_batch.png')
        # plt.show()

        # Concatenate tensors in each list along the batch dimension
        combined_depth1 = torch.cat(app1, dim=0).to(device)
        combined_depth2 = torch.cat(app2, dim=0).to(device)

        # Pass the combined tensors through the autoencoder
        combined_emb1, combined_map1 = self.auto_encoder_depth(combined_depth1.view(combined_depth1.size(0), 1, 128, 128))
        combined_emb2, combined_map2 = self.auto_encoder_depth(combined_depth2.view(combined_depth2.size(0), 1, 128, 128))

        # Split embeddings and depth maps back into original sequences
        depth_embs1 = self.split_combined_output(combined_emb1, app1)
        depth_embs2 = self.split_combined_output(combined_emb2, app2)

        depth_maps1 = self.split_combined_output(combined_map1.view(combined_map1.size(0), -1), app1)
        depth_maps2 = self.split_combined_output(combined_map2.view(combined_map2.size(0), -1), app2)

        depth_embs1 = [F.normalize(seq, p=2, dim=-1) for seq in depth_embs1]
        depth_embs2 = [F.normalize(seq, p=2, dim=-1) for seq in depth_embs2]

        # # Initialize lists to store outputs
        # depth_embs1, depth_maps1 = [], []
        # depth_embs2, depth_maps2 = [], []
        #
        # # Process each item in app1 and app2 separately
        # for depth1, depth2 in zip(app1, app2):
        #     depth1 = depth1.to(device)  # Move to GPU
        #     depth2 = depth2.to(device)
        #
        #     # Pass each item through the autoencoder separately
        #     emb1, map1 = self.auto_encoder_depth(depth1.view(depth1.size(0), 1, 384, 384))
        #     emb2, map2 = self.auto_encoder_depth(depth2.view(depth2.size(0), 1, 384, 384))
        #
        #     # Store results
        #     depth_embs1.append(emb1)
        #     depth_maps1.append(map1.view(map1.size(0), -1))  # Flatten
        #
        #     depth_embs2.append(emb2)
        #     depth_maps2.append(map2.view(map2.size(0), -1))  # Flatten
        #
        # # Normalize embeddings for each sequence
        # depth_embs1 = [F.normalize(seq, p=2, dim=-1) for seq in depth_embs1]
        # depth_embs2 = [F.normalize(seq, p=2, dim=-1) for seq in depth_embs2]
        #
        padded_keys_dict = self.feat_padding(app1, app2, depth_maps1,
                                                                     depth_maps2, depth_embs1, depth_embs2, device)

        # gt = []
        # for gt1, gt2 in zip(depth_embs1, depth_embs2):
        #     gt_matches = torch.zeros((len(gt1), len(gt2)), dtype=torch.float32)
        #     for i, id1 in enumerate(gt1):
        #         for j, id2 in enumerate(gt2):
        #             # Apply softmax to both id1 and id2
        #             id1_softmax = torch.nn.functional.softmax(id1, dim=-1)
        #             id2_softmax = torch.nn.functional.softmax(id2, dim=-1)
        # 
        #             # Check if the index of the max value matches
        #             if torch.argmax(id1_softmax) == torch.argmax(id2_softmax):
        #                 gt_matches[i, j] = 1
        #     gt.append(gt_matches)
        # gt = torch.stack(gt)
        # final_score = gt.cuda()

        
        final_score = torch.bmm(padded_keys_dict['depth_embs1'], padded_keys_dict['depth_embs2'].transpose(-2, -1)) #+ gt.cuda()

        return final_score, (padded_keys_dict['depth_embs1'], padded_keys_dict['depth_embs2'], padded_keys_dict['depth_maps1'], 
                             padded_keys_dict['app_feat1'], padded_keys_dict['depth_maps2'], padded_keys_dict['app_feat2'])
    

    def feat_padding(self, app_feat1, app_feat2, depth_maps1, depth_maps2, depth_embs1, depth_embs2, device):
        # Dictionary to store padded tensors and masks
        padded_keys_dict = {}
        # key_padding_masks_dict = {}

        # Determine the maximum sequence length in the batch
        max_seq_len = 0
        for key in app_feat1 + app_feat2:
            max_seq_len = max(max_seq_len, key.size(0))

        # Loop through each list to pad tensors and create masks
        for key_name, key_list in zip(
                ['app_feat1', 'app_feat2', 'depth_maps1', 'depth_maps2', 'depth_embs1', 'depth_embs2'],
                [app_feat1, app_feat2, depth_maps1, depth_maps2, depth_embs1, depth_embs2]):

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
            # key_padding_masks_dict[key_name] = key_padding_mask.to(device)

        return padded_keys_dict#, key_padding_masks_dict

    # Function to split combined outputs back into original sequences
    def split_combined_output(self, combined_output, original_list):
        split_sizes = [tensor.size(0) for tensor in original_list]
        return list(torch.split(combined_output, split_sizes, dim=0))


