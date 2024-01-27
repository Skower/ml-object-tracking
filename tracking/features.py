import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np


class DeepFeatures:
    """
    Compute similarity matrix between the current tracks and the new detections
    with generated features from a pretrained model.
    """

    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model = self.model.features  # Remove the classification head
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_bbox_patch(self, frame, bbox):
        """
        Extract the patch from the frame with the given bounding box.
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        return frame[y1:y2, x1:x2]

    def extract_features(self, img_patches):
        """
        Extract features from the given batch of image patches.
        """

        img_tensors = torch.stack(
            [
                self.preprocess(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
                for patch in img_patches if patch is not None and patch.size != 0
            ]
        )

        with torch.no_grad():
            features = self.model(img_tensors)

        features = features.view(features.size(0), -1)

        return features.cpu().numpy()

    def detections_to_patches(self, frame, detections):
        """
        Extract the patches from the frame with the given detections.
        """
        patches = []
        for detection in detections:
            patches.append(self.get_bbox_patch(frame, detection.bbox))

        return patches

    def similarity(self, frame, tracks, detections):
        """
        Compute the similarity matrix between the current tracks and the new detections
        with cosine similarity.
        """
        similarity = np.zeros((len(tracks), len(detections)))

        # Extract patches for all tracks and detections
        tracks_patches = self.detections_to_patches(frame, tracks)
        detections_patches = self.detections_to_patches(frame, detections)

        # Extract features for all tracks and detections in batches
        track_features = self.extract_features(tracks_patches)
        detection_features = self.extract_features(detections_patches)

        # Compute similarity matrix
        for i, track_feature in enumerate(track_features):
            for j, detection_feature in enumerate(detection_features):
                similarity[i, j] = torch.nn.functional.cosine_similarity(
                    torch.from_numpy(track_feature).unsqueeze(0), 
                    torch.from_numpy(detection_feature).unsqueeze(0), 
                    dim=1
                )

        return similarity