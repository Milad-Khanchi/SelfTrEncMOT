"""Generic detector."""
import os
import pickle

import torch

import yolox_adaptor

def detection_weights(dataset, test=False):

    if dataset:
        if test:
            detector_path = "./weights/bytetrack_x_mot17.pth.tar"
        else:
            detector_path = "./weights/bytetrack_ablation.pth.tar"
        size = (800, 1440)
    elif dataset == "mot20":
        if test:
            detector_path = "./weights/bytetrack_x_mot20.tar"
            size = (896, 1600)
        else:
            # Just use the mot17 test model as the ablation model for 20
            detector_path = "./weights/bytetrack_x_mot17.pth.tar"
            size = (800, 1440)
    elif dataset == "dance":
        # Same model for test and validation
        detector_path = "./weights/bytetrack_dance_model.pth.tar"
        size = (800, 1440)
    elif dataset == "sport":
        # Same model for test and validation
        detector_path = "./weights/SportsMOT_yolox_x_mix.tar"
        size = (800, 1440)
    else:
        raise RuntimeError("Need to update paths for detector for extra datasets.")

    return detector_path, size


class Detector(torch.nn.Module):
    K_MODELS = {"yolox"}

    def __init__(self, model_type, path, dataset):
        super().__init__()
        if model_type not in self.K_MODELS:
            raise RuntimeError(f"{model_type} detector not supported")

        self.model_type = model_type
        self.path = path
        self.dataset = dataset
        self.model = None

    def initialize_model(self):
        """Wait until needed."""
        if self.model_type == "yolox":
            self.model = yolox_adaptor.get_model(self.path, self.dataset)

    def forward(self, batch):
        if self.model is None:
            self.initialize_model()

        with torch.no_grad():
            batch = batch.half()
            output = self.model(batch)

        return output
