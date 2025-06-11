import numpy as np

from utils import *
from dotenv import load_dotenv
load_dotenv()

# Hyperparameters
batch_size = 1
det_thresh = 0.6

# Data transformation
transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
])

# Load the DanceTrack dataset
root_dir = os.getenv("DATASET_PATH", "datasets")  # fallback to "datasets" if not set
depth_feat_dir = './tracker-main/depth_frame_apple/'
app_feat_dir = './tracker-main/feat_embed/'
mask_feat_dir = './tracker-main/prompt_mask/'
test_dataset = DanceTrackDataset(root_dir=root_dir, depth_feat_dir=depth_feat_dir, app_feat_dir=app_feat_dir, mask_feat_dir=mask_feat_dir, split='val', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=10, pin_memory=True)

# Initialize the appearance model, depth model, and matching network

model = MatchingNetwork().cuda()

# Load the trained model weights
model.load_state_dict(torch.load('../train_attention_embedded_feat/runs/matching_network_epoch11.pth'))

# Set the model to evaluation mode
model.eval()

# TensorBoard writer
#writer = SummaryWriter(log_dir='runs/matching_network_test')




# sequence Tracklets:
tracked_objs = {}

# Global mapping of Tracklets across frames:
seq_tracklets = []

# Tracking
with torch.no_grad():
    # Automatically use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx_, (img2, app_feat2, depth_feat2, mask_feat2, motion_feat2, bbox2,  img2_path, original_size) in enumerate(tqdm(test_dataloader, desc="Testing")):
        frame_bboxes2 = bbox2[0].cpu().numpy()
        img2_path = img2_path[0]

        """
            Assign unique IDs to frame one objects
        """
        if img2_path.split('/')[-1] == '00000001.jpg':
            # assign new id
            bbox_id = 0
            seq_tracklets = []
            for i in range(frame_bboxes2.shape[0]):
                bbox_id = bbox_id + 1
                frame_bboxes2[i][0] = bbox_id
                trklt = Tracklets(img1=img2[0], bbox1=frame_bboxes2[i], app1=app_feat2[0][i],
                                  depth1=depth_feat2[0][i], mask1=mask_feat2[0][i][0],  motion1=motion_feat2[0][i], img1_path=img2_path)
                seq_tracklets.append(trklt)

            # Add frame to sequence Tracklets
            seq_key = img2_path.split('/')[-3]
            frame = img2_path.split('/')[-1]
            frame = int(frame.split('.')[-2])

            # Create a column with the frame number repeated for each row
            frame_column = np.full((frame_bboxes2.shape[0], 1), frame)

            # Add the new column to the original array
            frame_bboxes2_with_frame = np.hstack((frame_bboxes2, frame_column))

            if seq_key in tracked_objs:
                print("Error, Sequence exist!!\n")
            else:
                # Create a new list with the value for this key
                tracked_objs[seq_key] = frame_bboxes2_with_frame

            continue

        # previous frame features
        img1 = []
        app_feat1 = []
        depth_feat1 = []
        mask_feat1 = []
        motion_feat1 = []
        bbox1 = []

        for t, trk in enumerate(seq_tracklets):
            # get data of tracklets
            img1.append(seq_tracklets[t].get_img())  # img2
            app_feat1.append(seq_tracklets[t].get_appearance_emb())  # app_feat2
            depth_feat1.append(seq_tracklets[t].get_depth_emb())  # depth_feat2
            mask_feat1.append(seq_tracklets[t].get_mask())  # mask2
            motion_feat1.append(seq_tracklets[t].get_motion_emb())  # motion_feat2
            bbox1.append(seq_tracklets[t].get_bbox())  # bbox2

        img1 = torch.stack(img1)  # Stack img1 tensors
        app_feat1 = [torch.stack(app_feat1)]  # Stack app_feat1 tensors
        depth_feat1 = [torch.stack(depth_feat1)]  # Stack depth_feat1 tensors
        mask_feat1 = [torch.stack(mask_feat1)]  # Stack mask tensors
        motion_feat1 = [torch.stack(motion_feat1)]  # Stack motion_feat1 tensors
        bbox1 = np.vstack(bbox1)

        # divide masks of previous frame and current frame for frame2
        mask_feat2_crr = [tensor[0] for tensor in mask_feat2[0]]
        mask_feat2_crr = [torch.stack(mask_feat2_crr)]
        mask_feat2_prv = [tensor[1] for tensor in mask_feat2[0]]
        mask_feat2_prv = [torch.stack(mask_feat2_prv)]



        matched_ids_frame1 = set()
        with torch.cuda.amp.autocast():

            """
             First round of association
            """
            # Bytetrack idea
            bbox_scores2 = frame_bboxes2[:, 5]
            bboxes2 = frame_bboxes2[:, :5]
            high_score = bbox_scores2 > det_thresh
            high_score_bbox = bboxes2[high_score]
            high_det_score = bbox_scores2[high_score]

            #check whether there is any high_score_bbox
            if len(high_score_bbox) > 0:
                # divide objects based on score
                torch_high_score = torch.from_numpy(high_score).to(torch.bool)
                app_feat2_high = [tensor[torch_high_score] for tensor in app_feat2]
                depth_feat2_high = [tensor[torch_high_score] for tensor in depth_feat2]
                mask_feat2_prv_high = [tensor[torch_high_score] for tensor in mask_feat2_prv]
                mask_feat2_crr_high = [tensor[torch_high_score] for tensor in mask_feat2_crr]
                motion_feat2_high = [tensor[torch_high_score] for tensor in motion_feat2]

                iou_score, scores, _ = model(
                    app1=app_feat1,
                    app2=app_feat2_high,
                    depth1=depth_feat1,
                    depth2=depth_feat2_high,
                    mask1 = mask_feat1,
                    mask2=mask_feat2_prv_high,
                    motion1=motion_feat1,
                    motion2=motion_feat2_high,
                    device=device
                )

                # matching based on linear algorithm
                matcher = linear_assignment(scores)
                for bt, mat in enumerate(matcher):
                    for i, j in mat:
                        if (j >= high_score_bbox.shape[0]) or (i >= bbox1.shape[0]):
                            continue
                        if iou_score[bt][i][j].item() < 0.1:
                            continue
                        high_score_bbox[j][0] = bbox1[i][0]
                        matching_row_idx = np.where(np.all(frame_bboxes2[:, 1:5] == high_score_bbox[j, 1:], axis=1))[0]
                        matched_ids_frame1.add(bbox1[i][0])
                        frame_bboxes2[matching_row_idx[0], 0] = bbox1[i][0]
                        # update tracklets
                        seq_tracklets[i].update_feats(img1=img2[0], bbox1=frame_bboxes2[matching_row_idx[0]],
                                                      img1_path=img2_path, app1=app_feat2_high[0][j],
                                                      depth1=depth_feat2_high[0][j], mask1=mask_feat2_crr_high[0][j], motion1=motion_feat2_high[0][j], det_score=high_det_score[j])
                    # # plot
                    # visualize_matches(img2_path, bbox1, bboxes2, matcher[0],
                    #                     scores.squeeze(0).cpu().numpy())
            """
             Second round of association
            """
            # Find unmatched objects in frame1
            unmatch_inds = ~np.isin(bbox1[:, 0], list(matched_ids_frame1))
            bbox1 = bbox1[unmatch_inds]
            torch_mask = torch.from_numpy(unmatch_inds).to(torch.bool)
            app_feat1 = [tensor[torch_mask] for tensor in app_feat1]
            depth_feat1 = [tensor[torch_mask] for tensor in depth_feat1]
            mask_feat1 = [tensor[torch_mask] for tensor in mask_feat1]
            motion_feat1 = [tensor[torch_mask] for tensor in motion_feat1]

            # Find unmatched objects in frame2
            nan_indices = np.isnan(frame_bboxes2[:, 0])
            low_score_bbox = bboxes2[nan_indices]
            det_score = frame_bboxes2[:, 5]
            low_det_score = det_score[nan_indices]

            # check whether there is any low_score_bbox
            if len(low_score_bbox) > 0 and len(bbox1) > 0:
                # divide objects based on score
                torch_low_score = torch.from_numpy(nan_indices).to(torch.bool)
                app_feat2_low = [tensor[torch_low_score] for tensor in app_feat2]
                depth_feat2_low = [tensor[torch_low_score] for tensor in depth_feat2]
                mask_feat2_prv_low = [tensor[torch_low_score] for tensor in mask_feat2_prv]
                mask_feat2_crr_low = [tensor[torch_low_score] for tensor in mask_feat2_crr]
                motion_feat2_low = [tensor[torch_low_score] for tensor in motion_feat2]

                iou_score, scores, _ = model(
                    app1=app_feat1,
                    app2=app_feat2_low,
                    depth1=depth_feat1,
                    depth2=depth_feat2_low,
                    mask1=mask_feat1,
                    mask2=mask_feat2_prv_low,
                    motion1=motion_feat1,
                    motion2=motion_feat2_low,
                    device=device
                )

                # matching based on linear algorithm
                matcher = linear_assignment(scores)
                for bt, mat in enumerate(matcher):
                    for i, j in mat:
                        if (j >= low_score_bbox.shape[0]) or (i >= bbox1.shape[0]):
                            continue
                        if iou_score[bt][i][j].item() < 0.1:
                            continue
                        low_score_bbox[j][0] = bbox1[i][0]
                        matching_row_idx = np.where(np.all(frame_bboxes2[:, 1:5] == low_score_bbox[j, 1:], axis=1))[0]
                        matched_ids_frame1.add(bbox1[i][0])
                        frame_bboxes2[matching_row_idx[0], 0] = bbox1[i][0]
                        # update tracklets
                        m_r_idx = [
                            iter for iter, tracklet in enumerate(seq_tracklets)
                            if np.array_equal(tracklet.get_bbox(), bbox1[i])
                        ]
                        seq_tracklets[m_r_idx[0]].update_feats(img1=img2[0], bbox1=frame_bboxes2[matching_row_idx[0]],
                                                      img1_path=img2_path, app1=app_feat2_low[0][j],
                                                      depth1=depth_feat2_low[0][j], mask1=mask_feat2_crr_low[0][j], motion1=motion_feat2_low[0][j], det_score=low_det_score[j])
                    # plot
                    # visualize_matches(img2_path, bbox1, low_score_bbox, matcher[0],
                    #                     scores.squeeze(0).cpu().numpy())
                    print('changed')
        """
            Add new id to dict for unassigned objects
        """
        # Find unmatched objects in frame2
        nan_indices = np.isnan(frame_bboxes2[:, 0])
        left_score_bbox = bboxes2[nan_indices]

        # check whether there is any left_score_bbox
        if len(left_score_bbox) > 0:
            # divide objects based on score
            torch_left_score = torch.from_numpy(nan_indices).to(torch.bool)
            app_feat2_left = [tensor[torch_left_score] for tensor in app_feat2]
            depth_feat2_left = [tensor[torch_left_score] for tensor in depth_feat2]
            mask_feat2_left = [tensor[torch_left_score] for tensor in mask_feat2_crr]
            motion_feat2_left = [tensor[torch_left_score] for tensor in motion_feat2]

        for i in range(left_score_bbox.shape[0]):
            bbox_id = bbox_id + 1
            matching_row_idx = np.where(np.all(frame_bboxes2[:, 1:5] == left_score_bbox[i, 1:], axis=1))[0]
            frame_bboxes2[matching_row_idx[0]][0] = bbox_id
            trklt = Tracklets(img1=img2[0], bbox1=frame_bboxes2[matching_row_idx[0], :], app1=app_feat2_left[0][i],
                              depth1=depth_feat2_left[0][i], mask1=mask_feat2_left[0][i], motion1=motion_feat2_left[0][i], img1_path=img2_path)
            seq_tracklets.append(trklt)

        # Extract IDs
        ids = frame_bboxes2[:, 0]

        # Check for duplicates
        if len(ids) != len(np.unique(ids)):
            raise ValueError("Duplicate values found in IDs")

        """
            check age of tracklets, drop old ones
        """
        curr_age = img2_path.split('/')[-1]
        curr_age = int(curr_age.split('.')[-2])

        seq_tracklets = [trk for trk in seq_tracklets if trk.get_age() >= (curr_age - 15)]

        """
            Store objects with their ID
        """
        seq_key = img2_path.split('/')[-3]
        frame = img2_path.split('/')[-1]
        frame = int(frame.split('.')[-2])

        # Create a column with the frame number repeated for each row
        frame_column = np.full((frame_bboxes2.shape[0], 1), frame)

        # Removes targets not meeting threshold criteria
        frame_bboxes2_f = filter_targets(online_targets=frame_bboxes2)

        # Add the new column to the original array
        frame_bboxes2_with_frame = np.hstack((frame_bboxes2_f, frame_column))

        if seq_key in tracked_objs:

            # Append the value to the existing list for this key
            tracked_objs[seq_key] = np.vstack([tracked_objs[seq_key], frame_bboxes2_with_frame])
        else:
            # Create a new list with the value for this key
            tracked_objs[seq_key] = frame_bboxes2_with_frame

    """
        Store tracked objects in txt file
    """
    sort_and_save_tracked_objects(tracked_objs)

    """
        Post-processing tracked objects in txt file
    """
    dti("./test_attention_embedded_feat/results/DANCE-val/DANCETRACK/data","./test_attention_embedded_feat/results/post_processing/DANCETRACK")


print("Testing complete. Output images saved.")





