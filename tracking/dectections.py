import supervision as sv
from ultralytics import YOLO

class YoloDetector():
    """
    Dectector of pedestrians using YOLOv8.
    """

    def __init__(self):
        self.model = YOLO("yolov8n.pt", verbose=False)
        
        self.model.overrides["conf"] = 0.25  # NMS confidence threshold
        self.model.overrides["iou"] = 0.45  # NMS IoU threshold
        self.model.overrides["agnostic_nms"] = False  # NMS class-agnostic
        self.model.overrides["max_det"] = 500  # maximum number of detections per image
        self.model.overrides["device"] = "cpu"

    def detect(self, frame):
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Keep only pedestrians
        detections = detections[detections.class_id == 0]

        return detections

    def render(self, frame, detections):
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels = ["0" for class_id in detections.class_id]

        annotated_image = bounding_box_annotator.annotate(
            scene=frame, detections=detections
        )

        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        return annotated_image