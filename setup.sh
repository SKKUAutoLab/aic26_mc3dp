#!/bin/bash

# stop at the first error
set -e

# conda create --name Syn2RealTrack python=3.12 -y
# conda activate Syn2RealTrack

# Install the required packages.
pip install poetry==1.2.0  # Python dependency management and packaging made easy.
pip install pylabel==0.1.55  # A simple tool for labeling images for object detection.
pip install pycocotools==2.0.8  # Python API for MS COCO
pip install easydict==1.13  # A lightweight dictionary for Python.
pip install munch==4.0.0  # A dot-accessible dictionary (a la JavaScript objects).
pip install multipledispatch==1.0.0  # A generic function dispatcher in Python.
pip install rich==13.9.4  # Rich is a Python library for rich text and beautiful formatting in the terminal.
pip install pynvml==12.0.0  # Python utilities for the NVIDIA Management Library
pip install torchmetrics==1.6.1  # PyTorch native Metrics
pip install validators==0.34.0  # Python Data Validation for Humans
pip install thop==0.1.1.post2209072238  # A tool to count the FLOPs of PyTorch model.
pip install psutil==7.0.0  # Cross-platform lib for process and system monitoring in Python.
pip install filterpy==1.4.5  # Kalman filtering and optimal estimation library
pip install gdown==5.2.0  # Google Drive Public File/Folder Downloader
pip install tensorboard  # TensorFlow's Visualization Toolkit
pip install yacs==0.1.8  # Yet Another Configuration System
pip install termcolor==2.5.0  # ANSI color formatting for output in terminal
pip install lap==0.5.12  # Linear Assignment Problem solver (LAPJV/LAPMOD).
pip install einops==0.8.1  # A new flavour of deep learning operations
pip install opencv-python==4.12.0.88  # Open Source Computer Vision Library
pip install ordered-enum==0.0.10  # An ordered enum class for Python
pip install loguru==0.7.3 # Python logging made (stupidly) simple.
# pip install PyQt6==6.9.1 # Python bindings for the Qt cross-platform application and UI framework.
pip install shapely==2.1.2 # Manipulation and analysis of geometric objects
pip install tabulate==0.10.0 # Pretty-print tabular data
pip install h5py==3.16.0  # Read and write HDF5 files from Python

pip install transformers==5.9.0 # (HuggingFAce) Transformers: the model-definition framework for state-of-the-art machine learning models in text, vision, audio, and multimodal models, for both inference and training.
pip install supervision==0.28.0 # A set of easy-to-use utils that will come in handy in any Computer Vision project
pip install accelerate==1.13.0 # Accelerate training and inference of PyTorch models with multi-GPU, mixed-precision, and distributed training.
pip install timm==1.0.27 # A collection of image models for PyTorch, including vision transformers.
pip install albumentations==1.3.1 # Fast and flexible image augmentations
pip install omegaconf==2.3.0 # A hierarchical configuration system, with support for merging configurations from multiple sources and dynamic command-line overrides.
pip install pandas==3.0.3 # Powerful data structures for data analysis, time series, and statistics
pip install monai==1.5.2 # Medical Open Network for AI (MONAI) is a PyTorch-based framework for deep learning in healthcare imaging.
pip install deepdiff==9.1.0 # DeepDiff is a Python library that provides a way to compare complex data structures, such as dictionaries, lists, sets, and custom objects, and identify the differences between them.

pip install pydantic==2.13.4 # Data validation and settings management using Python type annotations.

pip install open3d==0.19.0  # A modern library for 3D data processing (point clouds)
pip install trimesh==4.12.2  # Import, export, and process triangular meshes
pip install pycolmap==4.0.4  # Python bindings for the COLMAP structure-from-motion pipeline
pip install plyfile==1.1.3  # PLY (Polygon File Format) reader and writer
pip install addict==2.4.0  # A dictionary whose items can be set using both attribute and item syntax
pip install evo==1.36.5  # Evaluation of odometry and SLAM (Umeyama alignment)
pip install scikit-learn==1.9.0  # Machine learning library (HDBSCAN / GaussianMixture clustering)

# Run line by line
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118


# In stall the project in editable mode.
rm -rf poetry.lock
poetry install --extras "dev"
rm -rf poetry.lock

echo "Finish installation successfully!"
