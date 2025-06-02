import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from fastreid_adaptor import FastReID

class app_feat_Extractor:
    def __init__(self, dataset, test=False):
        self.test_dataset = test
        self.dataset = dataset
        self.model = None
        self.crop_size = (128, 384)
        self.normalize = False

    def extract_features_app(self, img, bboxes):
        if self.model is None:
            self.initialize_model()

        # Generate all of the patches
        crops = []
        bbox = np.array(bboxes)
        results = bbox[:, 1:].astype(np.int32)
        h, w = img.shape[:2]
        results[:, 0] = results[:, 0].clip(0, w)
        results[:, 1] = results[:, 1].clip(0, h)
        results[:, 2] = results[:, 2].clip(0, w)
        results[:, 3] = results[:, 3].clip(0, h)

        crops = []
        for p in results:
            crop = img[p[1]: p[1] + p[3], p[0]: p[0] + p[2]]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            #plot
            # plt.imshow(crop.astype(np.uint8))  # Convert to uint8 for display
            # plt.savefig('./feat_embed/test_image_pairs_batch_{2}.png')
            # plt.show()
            if self.normalize:
                crop /= 255
                crop -= np.array((0.485, 0.456, 0.406))
                crop /= np.array((0.229, 0.224, 0.225))
            crop = torch.as_tensor(crop.transpose(2, 0, 1))
            crop = crop.unsqueeze(0)
            crops.append(crop)

        crops = torch.cat(crops, dim=0)
        embs = []
        for idx in range(0, len(crops), 1024):
            batch_crops = crops[idx: idx + 1024]
            batch_crops = batch_crops.cuda()
            with torch.no_grad():
                batch_embs = self.model(batch_crops)
            embs.extend(batch_embs)
        embs = torch.stack(embs)
        embs = torch.nn.functional.normalize(embs, dim=-1)
        embs = embs.cpu()

        features = []
        for i in range(results.shape[0]):
            features.append((embs[i], bbox[i]))

        return features  # embs and bbox

    def initialize_model(self):
        if self.dataset == "mot17":
            if self.test_dataset:
                path = "weights/mot17_sbs_S50.pth"
            else:
                return self._get_general_model()
        elif self.dataset == "mot20":
            if self.test_dataset:
                path = "weights/mot20_sbs_S50.pth"
            else:
                return self._get_general_model()
        elif self.dataset == "dance":
            path = "weights/dance_sbs_S50.pth"
        elif self.dataset == "sport":
            path = "weights/sports_sbs_S50.pth"
        else:
            raise RuntimeError("Need the path for a new ReID model.")

        model = FastReID(path)
        model.eval()
        model.cuda()
        model.half()
        self.model = model

    def _get_general_model(self):
        """Used for the half-val for MOT17/20.

        The MOT17/20 SBS models are trained over the half-val we
        evaluate on as well. Instead we use a different model for
        validation.
        """
        model = torchreid.models.build_model(name="osnet_ain_x1_0", num_classes=2510, loss="softmax", pretrained=False)
        sd = torch.load("weights/osnet_ain_ms_d_c.pth.tar")["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in sd.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()
        model.cuda()
        self.model = model
        self.crop_size = (128, 256)
        self.normalize = True










