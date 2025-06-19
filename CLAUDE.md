# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

House-GAN++ is a generative adversarial network for architectural floorplan generation and refinement. The project implements a GAN-based system that can generate realistic building layouts by learning from the RPLAN dataset of professional architectural drawings.

## Key Commands

### Running Inference
```bash
# Original inference (requires dataset)
python test.py

# User-friendly web interface
python run_ui.py

# Direct inference with architectural constraints
python inference.py
```
The web interface provides an intuitive way to generate floorplans with architectural intelligence.

### Training Models
```bash
python train.py --data_path /path/to/dataset --target_set 8
```
Trains the GAN models. Checkpoints are saved to `./checkpoints/` and training visualizations to `./exps/`.

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

## Architecture Overview

### Core Components

**Models** (`models/`):
- `models.py`: Contains the main Generator and Discriminator classes
- `models_improved.py`: Enhanced model variants
- `model_resnet.py`: ResNet-based architectures

**Dataset** (`dataset/`):
- `floorplan_dataset_maps_functional_high_res.py`: Main dataset loader for RPLAN data
- Handles JSON floorplan files with room types, bounding boxes, and edge information

**Training Pipeline**:
- `train.py`: Main training script with WGAN-GP loss
- `test.py`: Inference script for generating floorplans
- Supports incremental room generation by type

### Key Data Flow

1. **Input**: JSON files containing room types, bounding boxes, and connectivity graphs
2. **Processing**: Graphs are converted to node/edge representations with spatial masks
3. **Generation**: Generator creates room layouts conditioned on graph structure
4. **Refinement**: Iterative generation allows for progressive layout development

### Utilities

**Misc** (`misc/`):
- `utils.py`: Core utilities for graph processing, mask drawing, and data initialization
- `read_data.py`, `read_floorplan.py`: Data loading utilities
- `compute_FID.py`: Evaluation metrics

**Testing** (`testing/`):
- Various evaluation scripts for metrics like FID, GED
- Visualization and reconstruction utilities

## Data Format

The system expects JSON files with:
- `room_type`: List of room type IDs
- `boxes`: Bounding box coordinates for each room
- `edges`: Wall/boundary definitions
- `ed_rm`: Edge-to-room mappings

## Model Checkpoints

Pretrained models are stored in `checkpoints/pretrained.pth`. Training saves intermediate checkpoints as `{exp_folder}_{iteration}.pth`.

## User Interface

### Web Interface
- `web_interface.py`: Streamlit-based web application with architectural intelligence
- `inference.py`: Architectural inference engine with professional design constraints
- `run_ui.py`: Launcher script that provides system checks and easy startup

The interface uses architectural templates and adjacency rules to generate realistic, professional-quality floorplans that respect building design principles.