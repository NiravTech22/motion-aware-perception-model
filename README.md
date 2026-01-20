# Motion Aware Perception Model

[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-orange)](https://developer.nvidia.com/cuda-zone)
[![C++](https://img.shields.io/badge/C%2B%2B-ISO%20%2B%2B20-blue)](https://isocpp.org/)


High-performance GPU-accelerated perception pipeline for autonomous robotic systems, designed for real-time deployment and simulation-to-real workflows.

---

## Overview

This repository implements a custom-trained perception model and inference pipeline optimized with CUDA acceleration for robotics and autonomous systems. It supports a hybrid simulation environment (e.g., Isaac Sim), as well as real robotic hardware, to enable end-to-end perception from sensor input to actionable outputs.

**Primary goals:**

- Real-time perception using GPU acceleration  
- Custom model training (no reliance on off-the-shelf perception stacks)  
- Robotics-ready integration (ROS / embedded pipelines)  
- Simulation-to-real transfer support  

---

## System Architecture (High-Level)

```mermaid
flowchart TD
    Camera["Camera / RGB-D / LiDAR"]
    SimFeed["Simulation Feed (Isaac Sim)"]
    PreNorm["Normalization (CUDA)"]
    PreResize["Resize / Crop"]
    Model["Custom Perception Model"]
    Filter["Noise Filtering (CUDA)"]
    Format["Format to ROS / Control"]
    ROSNode["ROS2 Node / Embedded Control"]
    Actuation["Actuator Commands"]

    Camera --> PreNorm
    SimFeed --> PreNorm
    PreNorm --> PreResize --> PreAug --> Model
    Model --> TensorRT --> Filter --> Format --> ROSNode --> Actuation
```
---
