# CARLA Synthetic Perception Dataset & YOLOv8 Object Detector

## Overview

A synthetic dataset generation pipeline for CARLA 0.9.15 that produces
depth-verified, occlusion-aware ground-truth labels (vehicle / pedestrian)
for 2D object detection, paired with a YOLOv8 model trained on the
resulting data.

The goal is to build an end-to-end perception pipeline — from synthetic
data generation, through object detection training, to evaluation and
(in later phases) real-time deployment.

The dataset currently contains two classes:

- Vehicle
- Pedestrian

---

## Phase 1 — Baseline Dataset and YOLOv8 Training

### Dataset

The dataset was generated entirely within the CARLA simulation
environment, using a custom pipeline built for this project:

CARLA Simulation
↓
RGB + Depth Camera Capture (synchronized)
↓
3D Actor Bounding Boxes
↓
2D Bounding Box Projection
↓
Depth-Based Visibility Filtering (occlusion-aware)
↓
YOLO-Format Dataset
↓
YOLOv8 Training

- **Image resolution:** 1280 × 720
- **Dataset size:** 2,000 labeled frames (1,800 train / 200 validation)
- **Classes:** vehicle, pedestrian

A depth-based visibility filter determines whether each actor's
projected bounding box should be kept, based on whether the actor is
fully visible, partially occluded, or fully occluded by another object
in the scene — rather than relying on 2D projection alone.

### Model

The baseline model is **YOLOv8 Nano (`yolov8n.pt`)**, fine-tuned from
pretrained COCO weights on the synthetic CARLA dataset.

### Training Duration Experiment

Three training durations were tested at a fixed image size (1280) to
identify where the model converges:

| Epochs | Precision | Recall | mAP50 | mAP50-95 |
|-------:|----------:|-------:|------:|---------:|
| 50     | 0.915     | 0.812  | 0.869 | 0.649    |
| 100    | 0.924     | 0.825  | 0.881 | 0.692    |
| 150    | 0.943     | 0.821  | 0.887 | 0.693    |

**Results by class:**

**Vehicle**

| Epochs | mAP50 | mAP50-95 |
|-------:|------:|---------:|
| 50     | 0.954 | 0.798    |
| 100    | 0.952 | 0.816    |
| 150    | 0.961 | 0.826    |

**Pedestrian**

| Epochs | mAP50 | mAP50-95 |
|-------:|------:|---------:|
| 50     | 0.784 | 0.500    |
| 100    | 0.810 | 0.568    |
| 150    | 0.813 | 0.560    |

### Key Findings

- Overall performance improved substantially from 50 to 100 epochs.
- From 100 to 150 epochs, overall mAP50-95 was essentially flat
  (0.692 → 0.693), and pedestrian recall and mAP50-95 both slightly
  **regressed** (0.737 → 0.718 recall; 0.568 → 0.560 mAP50-95). This is
  consistent with the model continuing to optimize for the majority
  class (vehicle, ~5:1 instance ratio over pedestrian) once the
  pedestrian class had already converged.
- Vehicle detection is strong across all durations tested.
- Pedestrian detection is the harder class, both in recall and in
  localization precision — likely driven by class imbalance and the
  smaller pixel footprint of pedestrian instances.
- Based on this, **100 epochs was selected as the Phase 1 baseline**,
  giving the best trade-off between performance and training cost
  before returns diminished (and slightly reversed for pedestrians).

### Phase 1 Status

- [x] CARLA simulation environment set up
- [x] Synchronized RGB + depth capture
- [x] 3D → 2D bounding box projection
- [x] Depth-based, occlusion-aware visibility filtering
- [x] YOLO-format dataset generation
- [x] YOLOv8 training
- [x] Training duration / convergence experiment
- [x] Per-class evaluation and analysis

### Next Steps

Future phases will focus on dataset diversity and detection
performance, including:

- More diverse weather and lighting conditions
- Additional CARLA maps
- More pedestrian samples, to address the current class imbalance
- Small/distant object analysis
- Model export and inference optimization (ONNX / TensorRT)
- ROS2 integration for real-time perception

### Phase 1 Conclusion

A complete baseline pipeline was built for generating synthetic CARLA
data and training a YOLOv8 detector on it, including a controlled
experiment on training duration to identify convergence. The results
give a solid baseline for future work on generalization and, in
particular, pedestrian detection performance.

### Roadmap

This is Phase 1 of a larger perception pipeline project. Planned future
phases include model export/optimization (ONNX, TensorRT), sensor
fusion with LiDAR, object tracking, and ROS2 integration for real-time
perception. Each phase will be documented here as it's completed.