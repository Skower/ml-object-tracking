import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

from .features import DeepFeatures
from .KalmanFilter import KalmanFilter

def xyxy_to_xywh(box):
    """
    Convert the bbox format from [x1, y1, x2, y2] to [x1, y1, width, height].
    """
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]

class Detection:
    def __init__(
        self,
        bbox,
        identifier,
        confidence,
        dt=1.0,
        u_x=0.0,
        u_y=0.0,
        std_acc=1.0,
        x_std_meas=1.0,
        y_std_meas=1.0,
    ):
        self.bbox = bbox
        self.identifier = identifier
        self.confidence = confidence
        self.kalman_filter = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
        self.age = 1

        # Calculate the center of the bbox
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2

        self.kalman_filter = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
        # Initialize Kalman filter state with bbox center position and zero velocity
        self.kalman_filter.x = np.matrix([[x_center], [y_center], [0], [0]])

    def miss(self):
        """
        Decrement the age of the track.
        """
        self.age -= 1

    def __repr__(self):
        return f"Detection(bbox={self.bbox}, identifier={self.identifier}, confidence={self.confidence})"


def format_detection(detections, identifier, confidence_threshold=0.2):
    """
    Convert the Supervision detections to the format used by the tracker.
    """
    formatted_detections = []

    for i in range(len(detections.xyxy)):
        if detections.confidence[i] < confidence_threshold:
            continue

        bbox = detections.xyxy[i]
        confidence = detections.confidence[i]
        detection = Detection(bbox, identifier, confidence)
        formatted_detections.append(detection)
        identifier += 1

    return formatted_detections, identifier


class Tracker:
    def __init__(self, initial_detections):
        self.tracks, self.identifier = format_detection(initial_detections, 1)
        self.deep_features = DeepFeatures()

    def _IoU(self, box1, box2):
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.
        """
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        x5, y5 = max(x1, x3), max(y1, y3)
        x6, y6 = min(x2, x4), min(y2, y4)

        intersection = max(0, x6 - x5 + 1) * max(0, y6 - y5 + 1)

        area1 = (x2 - x1 + 1) * (y2 - y1 + 1)
        area2 = (x4 - x3 + 1) * (y4 - y3 + 1)

        iou = intersection / float(area1 + area2 - intersection)
        return iou

    def IoU_similarity(self, dectections):
        """
        Compute the similarity between the current tracks and the new detections.
        """
        iou = np.zeros((len(self.tracks), len(dectections)))
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(dectections):
                iou[i, j] = self._IoU(track.bbox, detection.bbox)

        return iou

    def similarity(self, frame, detections, size_threshold=0.5, threshold=0.3, alpha=0.5, beta=0.5):
        similarity_matrix = alpha * self.IoU_similarity(
            detections
        ) + beta * self.deep_features.similarity(frame, self.tracks, detections)

        # Filter out the detections that are too different in size
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                track_size = (track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1])
                detection_size = (detection.bbox[2] - detection.bbox[0]) * (detection.bbox[3] - detection.bbox[1])
                size_ratio = track_size / detection_size if detection_size > 0 else 0
                if size_ratio < size_threshold or size_ratio > 1 / size_threshold:
                    similarity_matrix[i, j] = 0

        similarity_matrix[similarity_matrix < threshold] = 0

        return similarity_matrix

    def update(self, frame, detections, frame_index):
        """
        Update the tracks with the new detections.

        IoU > 0.5: same object

        1. Update the tracks that match the new detections and keep same identifier. (linear assignment)
        2. Remove the tracks that do not match any new detections.
        3. Add new tracks for the new detections.
        """
        detections, _ = format_detection(
            detections, self.identifier, confidence_threshold=0.4
        )
        similarity = self.similarity(frame, detections)
 
        matched_indices = linear_sum_assignment(similarity, maximize=True)
        matched_indices = np.asarray(matched_indices)

        # Update the tracks that match the new detections and keep same identifier.
        for i in range(len(matched_indices[0])):
            track_index = matched_indices[0][i]
            detection_index = matched_indices[1][i]

            if similarity[track_index, detection_index] > 0:
                previous_identifier = self.tracks[track_index].identifier
                self.tracks[track_index] = detections[detection_index]
                self.tracks[track_index].identifier = previous_identifier
                self.tracks[track_index].age += 1

                new_detection = detections[detection_index]

                # Predict the state of the track
                self.tracks[track_index].kalman_filter.predict()

                # Calculate the center of the new bbox for the measurement update
                new_x_center = (new_detection.bbox[0] + new_detection.bbox[2]) / 2
                new_y_center = (new_detection.bbox[1] + new_detection.bbox[3]) / 2
                measurement = np.matrix([[new_x_center], [new_y_center]])

                # Update Kalman Filter with the new detection center
                update_state = self.tracks[track_index].kalman_filter.update(
                    measurement
                )
                x_center, y_center = update_state[0, 0], update_state[1, 0]
                width, height = (
                    new_detection.bbox[2] - new_detection.bbox[0],
                    new_detection.bbox[3] - new_detection.bbox[1],
                )
                x1, y1 = x_center - width / 2, y_center - height / 2
                x2, y2 = x_center + width / 2, y_center + height / 2
                self.tracks[track_index].bbox = [x1, y1, x2, y2]

        # Remove the tracks that do not match any new detections.
        unmatched_tracks = np.where(similarity.sum(axis=1) == 0)[0]
        for i in reversed(unmatched_tracks):
            # Age strategy to remove the track
            self.tracks[i].miss() 
            if self.tracks[i].age <= -3 or frame_index < 5:  
                self.tracks.pop(i)
          

        # Add new tracks for the new detections.
        unmatched_detections = np.where(similarity.sum(axis=0) == 0)[0]
        for i in unmatched_detections:
            detections[i].identifier = self.identifier
            detections[i].age = 1
            self.tracks.append(detections[i])
            self.identifier += 1

        return self.tracks

    def render_tracks(self, frame):
        """
        Render the tracks on the frame.
        """
        for track in self.tracks:
            bbox = track.bbox
            identifier = track.identifier
            confidence = track.confidence

            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(
                frame,
                f"{identifier} {confidence:.2f}",
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return frame
