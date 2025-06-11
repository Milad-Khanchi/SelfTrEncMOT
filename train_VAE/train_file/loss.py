import torch
import torch.nn as nn
from util import linear_assignment
import torch.nn.functional as F

class mseCorrespondingLoss(nn.Module):
    def __init__(self):
        super(mseCorrespondingLoss, self).__init__()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.MSELoss = nn.MSELoss()

    def forward(self, final_score, embs, gt_matches, epoch):
        (depth_emb1, depth_emb2, depth1_out, depth1, depth2_out, depth2) = embs
        # () = final_score
        total_loss = 0
        loss_count = 0

        for emb1, emb2, gt in zip(depth_emb1, depth_emb2, gt_matches):
            # Find indices where gt is 1
            matching_indices = torch.nonzero(gt > 0, as_tuple=False)

            # Extract corresponding embeddings
            emb1 = emb1[matching_indices[:, 0]]
            emb2 = emb2[matching_indices[:, 1]]

            # Compute MSE loss between matched embeddings
            if emb1.size(0) > 0:
                total_loss += self.MSELoss(emb1, emb2)
                loss_count += 1

        if epoch < 10:
            for emb1, emb2 in zip(depth1_out, depth1):
                total_loss += self.MSELoss(emb1, emb2)

            for emb1, emb2 in zip(depth2_out, depth2):
                total_loss += self.MSELoss(emb1, emb2)

        # Return the mean loss if there were valid pairs, otherwise return 0
        return total_loss / loss_count