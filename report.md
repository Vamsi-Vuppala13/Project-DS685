# Assignment 4 — Visual SLAM Report

## Visual SLAM from First Principles and Indoor Space Mapping

This repository contains my completed work for Robotics Assignment 4. The assignment focuses on implementing a Visual SLAM / Visual Odometry pipeline from first principles and then applying it to a real indoor video recorded by me.

The project includes feature detection, feature matching, pose estimation, trajectory generation, and visualization of the camera path.

---

## Task 1 — Visual SLAM from First Principles

### Objective

The first task was to build a basic Visual SLAM pipeline using Python without depending on a complete pre-built SLAM system. The goal was to understand the main steps involved in estimating camera motion from image frames.

### Implementation

The Visual SLAM pipeline was implemented using the following components:

| Component | Description |
|---|---|
| ORB Features | Used to detect and describe keypoints in each frame |
| Feature Matching | Used KNN matching with Lowe’s ratio test |
| Lowe’s Ratio Threshold | `0.75` |
| Essential Matrix | Estimated using RANSAC to remove weak/outlier matches |
| Pose Recovery | Rotation and translation were recovered from the Essential Matrix |
| Triangulation | Sparse 3D points were generated from matched 2D feature points |

### Key Functions

| Function | Purpose |
|---|---|
| `detect()` | Detects ORB keypoints and computes descriptors |
| `match()` | Matches features between two frames |
| `estimate_pose()` | Estimates the Essential Matrix and recovers camera pose |
| `process()` | Runs the full visual odometry pipeline frame by frame |

### Task 1 Results

The pipeline was tested on 50 synthetic KITTI-like frames. The system was able to estimate the camera trajectory across the frame sequence and generate a trajectory plot.

Output file:

```text
trajectory.png

## Task 2 — Visual SLAM on My Indoor Space

### Objective

The second task was to apply the Visual SLAM pipeline to my own indoor space video. The goal was to test whether the ORB-based visual odometry approach could estimate a camera trajectory from a real video instead of only using synthetic or dataset images.

### Input Video

For this task, I used an indoor room video recorded from a moving camera.

**YouTube Demo:**

```text
https://youtube.com/shorts/1eDGgk2OY3o
