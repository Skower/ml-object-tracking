from glob import glob

import cv2 as cv
from tracking import Tracker, YoloDetector, xyxy_to_xywh
import supervision as sv
import numpy as np

def read_trackeval_format(filename, frame_id) -> sv.Detections:
    """ Detections for 1 frame in MOTChallenge format"""
    bboxes = []
    confidences = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data = line.strip().split(',')
            frame, id, bb_left, bb_top, bb_width, bb_height, conf = map(float, data[:7])
            if frame == frame_id:
                bbox = [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height]
                bboxes.append(bbox)
                confidences.append(conf)

    bboxes = np.array(bboxes)
    confidences = np.array(confidences)

    class_id = np.ones(len(bboxes))
    frame = np.ones(len(bboxes)) * frame
    return sv.Detections(confidence=confidences, xyxy=bboxes, class_id=class_id)

if __name__ == "__main__":
    with open("tracking-results/ADL-Rundle-6.txt", "w") as f:
        detector = YoloDetector()
        
        frame1 = cv.imread("ADL-Rundle-6/img1/0001.jpg")
        initial_detections = detector.detect(frame1)

        tracker = Tracker(initial_detections)

        
        # Write the results to a text file for MOT evaluation
        for track in tracker.tracks:
            bbox = xyxy_to_xywh(track.bbox)
            f.write(
                f"1,{track.identifier},{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])},{track.confidence},-1,-1,-1\n"
            )
   
        for i in range(2, 526):
            frame = cv.imread(glob(f"ADL-Rundle-6/img1/0*{i}.jpg")[0])
            detections = detector.detect(frame)
      
            tracker.update(frame, detections, i)

            # Check that track IDs are unique
            assert len(set([track.identifier for track in tracker.tracks])) == len(
                tracker.tracks
            ), f"Track IDs are not unique: {[track.identifier for track in tracker.tracks]}"

            # Write the results to a text file for MOT evaluation
            for track in tracker.tracks:
                bbox = xyxy_to_xywh(track.bbox)
                f.write(
                    f"{i},{track.identifier},{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])},{track.confidence},-1,-1, -1\n"
                )

            annotated_image = tracker.render_tracks(frame)

            annotated_image = cv.resize(annotated_image, (0, 0), fx=0.5, fy=0.5)
            cv.imshow("MyTracker", annotated_image)

            if cv.waitKey(1) == ord("q"):
                break

        cv.destroyAllWindows()
