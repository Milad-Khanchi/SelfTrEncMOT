from util import *
from loss import *

import os

from dotenv import load_dotenv
load_dotenv()

# Hyperparameters
lr = 1e-3
batch_size = 64  # Reduce batch size
epochs = 15

# Data transformation
transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
])

# Load the DanceTrack dataset
root_dir = os.getenv("DATASET_PATH", "datasets")  # fallback to "datasets" if not set
depth_feat_dir = './tracker-main/depth_frame_apple/'
app_feat_dir = './tracker-main/prompt_mask/'

train_dataset = DanceTrackDataset(root_dir=root_dir, depth_feat_dir=depth_feat_dir, app_feat_dir=app_feat_dir,
                                  split='train', transform=transform)
val_dataset = DanceTrackDataset(root_dir=root_dir, depth_feat_dir=depth_feat_dir, app_feat_dir=app_feat_dir,
                                split='val', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

# Initialize the matching network
match_model = MatchingNetwork().cuda()
matching_loss = mseCorrespondingLoss()

# Optimizer
optimizer = optim.Adam(match_model.parameters(), lr=lr)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
scaler = torch.cuda.amp.GradScaler()

# TensorBoard writer
writer = SummaryWriter(log_dir='runs/matching_network_experiment')


# Training
for epoch in range(epochs):
    if epoch == 10:
        lr /= 10

    match_model.train()
    train_loss = 0
    for batch_idx, (
        img1, img2,
        app_feat1_batch, app_feat2_batch,
        motion_feat1_batch, motion_feat2_batch,
        bboxes1, bboxes2,
        gt_matches, path1_batch, path2_batch, original_size_batch
    ) in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # Forward pass
            # Automatically use GPU if available, otherwise fallback to CPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            scores, embs = match_model(
                                        app1=app_feat1_batch,
                                        app2=app_feat2_batch,
                                        motion1=motion_feat1_batch,
                                        motion2=motion_feat2_batch,
                                        device=device
                                        )
            # Compute loss
            loss = matching_loss(scores, embs, gt_matches.to(device), epoch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(match_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        # Save image pairs with annotations every 100 batches
        if batch_idx % 1500 == 0:
            matcher = linear_assignment(scores)
            matches = []
            for k, mat in enumerate(matcher):
                match = {}
                for i, j in mat:
                    # Check if both indices exist in bboxes1 and bboxes2
                    if k < len(bboxes1) and i < len(bboxes1[k]) and len(bboxes1[k][i]) > 0 and \
                            k < len(bboxes2) and j < len(bboxes2[k]) and len(bboxes2[k][j]) > 0:

                        # Check if the last element exists and is not None for both lists
                        if bboxes1[k][i][0] is not None and bboxes2[k][j][0] is not None:
                            # Perform the desired operation
                            match[bboxes1[k][i][0].item()] = bboxes2[k][j][0].item()
                matches.append(match)
            save_image_pairs(img1, img2, bboxes1, bboxes2, matches, epoch, batch_idx, original_size_batch, writer, mode='train')

    avg_train_loss = train_loss / len(train_dataloader)
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}')

    # Validation loop
    match_model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_idx, (
        img1, img2,
        app_feat1_batch, app_feat2_batch,
        motion_feat1_batch, motion_feat2_batch,
        bboxes1, bboxes2,
        gt_matches, path1_batch, path2_batch, original_size_batch
    ) in enumerate(
                tqdm(val_dataloader, desc="Val Loss")):
            with torch.cuda.amp.autocast():
                # Forward pass
                scores, embs = match_model(
                                        app1=app_feat1_batch,
                                        app2=app_feat2_batch,
                                        motion1=motion_feat1_batch,
                                        motion2=motion_feat2_batch,
                                        device=device
                                        )
                # Compute loss
                loss = matching_loss(scores, embs, gt_matches.to(device), epoch)
                val_loss += loss.item() if torch.isfinite(loss) else 0

                # Save image pairs with annotations every 100 batches
                if val_idx % 600 == 0:
                    matcher = linear_assignment(scores)
                    matches = []
                    for k, mat in enumerate(matcher):
                        match = {}
                        for i, j in mat:
                            # Check if both indices exist in bboxes1 and bboxes2
                            if k < len(bboxes1) and i < len(bboxes1[k]) and len(bboxes1[k][i]) > 0 and \
                                    k < len(bboxes2) and j < len(bboxes2[k]) and len(bboxes2[k][j]) > 0:

                                # Check if the last element exists and is not None for both lists
                                if bboxes1[k][i][0] is not None and bboxes2[k][j][0] is not None:
                                    # Perform the desired operation
                                    match[bboxes1[k][i][0].item()] = bboxes2[k][j][0].item()
                        matches.append(match)
                    save_image_pairs(img1, img2, bboxes1, bboxes2, matches, epoch, val_idx, original_size_batch,
                                     writer, mode='val')
                    torch.save(match_model.state_dict(),
                               f'./train_attention_embedded_feat/runs/weights/matching_network.pth')

    avg_val_loss = val_loss / len(val_dataloader)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    print(f'Epoch {epoch + 1}, Val Loss: {avg_val_loss}')

# Close the TensorBoard writer
writer.close()

# Save the trained matching network
torch.save(match_model.state_dict(), './train_attention_embedded_feat/runs/weights/matching_network_final.pth')


