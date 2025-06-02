from tqdm import tqdm
import os
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DanceTrackDataset, collate_fn

from detector import detector
from detector.detector import detection_weights

from dotenv import load_dotenv
load_dotenv()



# Argument parser setup
parser = argparse.ArgumentParser(description="YOLOX detection for MOT datasets.")
parser.add_argument("--dataset", type=str, default="dance", choices=["mot17", "mot20", "dance", "sport"], help="Dataset name (default: 'dance')")
parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split (default: 'val')")

args = parser.parse_args()

split = args.split
batch_size = 1
output_dir = './detection/detected_objs/'

dataset = args.dataset
yolo_weight, img_size = detection_weights(dataset=dataset, test=False)

# Data transformation
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

# Load the dataset
load_dotenv()
root_dir = os.getenv("DATASET_PATH", "datasets")  # fallback to "datasets" if not set
test_dataset = DanceTrackDataset(root_dir=root_dir, split=split, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

#Detector
detector = detector.Detector("yolox", yolo_weight, dataset)

previous_bboxes = []
previous_image_path = None
for idx, (img, frame, img_path, original_size) in enumerate(tqdm(test_dataloader, desc="detecting")):

    if previous_image_path is None:
        path_parts = img_path.split('/')
        previous_image_path = img_path.split('/')[-3]
        output_dir = os.path.join(output_dir, '/'.join(path_parts[-5:-3]))
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # Create the directory if it doesn't exist

    # Check if we have reached the next directory
    current_directory = img_path.split('/')[-3]
    if current_directory != previous_image_path:
        # Save previous bboxes to a txt file
        if previous_bboxes:
            output_path = os.path.join(output_dir, f'{previous_image_path}.txt')
            with open(output_path, 'w') as f:
                for bbox in previous_bboxes:
                    f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]},{bbox[5]}, {bbox[6]}\n")
            print(f"Saved bounding boxes to {output_path}")

        # Reset the previous bboxes list
        previous_bboxes = []
        previous_image_path = current_directory

    # Make predictions
    pred = detector(img.cuda())

    if pred is not None:
        # Rescale bounding boxes to original size
        bboxes = pred[:, :-1].cpu().numpy()  # Assuming the output is in the form of bounding boxes
        scores = pred[:, -1].cpu().numpy()  # Assuming scores are also returned
        original_width, original_height = original_size  # Get original image size

        # Rescale bounding boxes to the original image size
        rescaled_bboxes = []
        for bbox, score in zip(bboxes, scores):
            xmin, ymin, xmax, ymax = bbox
            xmin = int(xmin * original_width / img.shape[-1])  # Rescale according to original width
            xmax = int(xmax * original_width / img.shape[-1])
            ymin = int(ymin * original_height / img.shape[-2])  # Rescale according to original height
            ymax = int(ymax * original_height / img.shape[-2])

            # Convert to width and height format
            width = xmax - xmin
            height = ymax - ymin

            rescaled_bboxes.append([frame, 1, xmin, ymin, width, height, score])

        # Append to the previous_bboxes list
        previous_bboxes.extend(rescaled_bboxes)



# If any bboxes are left for the last directory, save them
if previous_bboxes:
    output_path = os.path.join(output_dir, f'{previous_image_path}.txt')
    with open(output_path, 'w') as f:
        for bbox in previous_bboxes:
            f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]},{bbox[5]}, {bbox[6]}\n")
    print(f"Saved bounding boxes to {output_path}\n")



print('done')