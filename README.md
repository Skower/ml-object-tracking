# Multi-Object Tracker

Authors: Bastien Pouëssel

This project is a multi-object tracker with several improvements, including the use of the YOLOv8 detector and an age-based approach for improved track matching.

## Project Architecture

The project is structured as follows:

- `main.py`: This is the main entry point of the application.
- `requirements.txt`: This file lists the Python dependencies required by the project.
- `Makefile`: This file contains commands for building and running the project.
- `tracking/`: This directory contains the core tracking logic.
  - `KalmanFilter.py`: Implements the Kalman Filter for object tracking.
  - `detections.py`: Contains the `YoloDetector` class which uses YOLOv8 for object detection.
  - `tracker.py`: Contains the `Tracker` class which handles the tracking of multiple objects.
  - `features.py`: CNN based approach for feature embedding

## How to Run

1. Install the required Python dependencies:

```sh
pip install -r requirements.txt
```

2. Build and run the project

```sh
make
```

The sequence should be in the directory ADL-Rundle-6 and the directory tracking-results should exists in order to produce the result for trackeval.