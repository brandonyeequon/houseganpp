# House-GAN++ Web Interface

A simple web interface for generating architectural floorplans using the House-GAN++ pretrained model.

## Features

- **Interactive Room Selection**: Choose from 12 different room types
- **Real-time Generation**: Generate floorplans in seconds
- **Visual Feedback**: Color-coded room legend and preview
- **Download Results**: Save generated floorplans as PNG images
- **Adjustable Settings**: Control refinement iterations and random seed

## Quick Start

### Prerequisites

1. Make sure you have the pretrained model at `./checkpoints/pretrained.pth`
2. Install required dependencies:

```bash
# Install PyTorch and other dependencies
pip install torch torchvision matplotlib numpy pillow opencv-python networkx svgwrite webcolors

# Install Streamlit for the web interface
pip install streamlit
```

### Launch the Interface

#### Option 1: Using the launcher script (recommended)
```bash
python run_ui.py
```

#### Option 2: Direct Streamlit command
```bash
streamlit run web_interface.py
```

The interface will open in your default web browser at `http://localhost:8501`

### Check System Requirements
```bash
python run_ui.py --check
```

## How to Use

1. **Select Room Types**: In the sidebar, check the boxes for rooms you want in your floorplan
2. **Adjust Settings** (optional): Expand "Advanced Settings" to modify:
   - Refinement iterations (1-10)
   - Random seed for reproducible results
3. **Generate**: Click "Generate Floorplan" button
4. **View Results**: The generated floorplan appears on the right side
5. **Download**: Click "Download Floorplan" to save the image

## Available Room Types

- Living Room
- Kitchen
- Bedroom
- Bathroom
- Balcony
- Entrance
- Dining Room
- Study Room
- Storage

## Technical Details

### Files Created

- `inference.py`: Architectural inference engine with professional design constraints
- `web_interface.py`: Streamlit web application with architectural intelligence
- `run_ui.py`: Launcher script with system checks

### How It Works

1. **Room Selection**: User selects desired room types from the interface
2. **Graph Creation**: The system creates a simple graph with all rooms connected
3. **Iterative Generation**: The model generates room layouts through multiple refinement iterations
4. **Visualization**: Generated masks are converted to colored room layouts

### Limitations

- Uses simplified graph connectivity (all rooms connected)
- Limited to predefined room types
- Generates fixed 256x256 pixel images
- No manual room positioning or sizing controls

## Troubleshooting

### Model Loading Issues
- Ensure `./checkpoints/pretrained.pth` exists
- Check that PyTorch is properly installed

### Import Errors
- Install missing dependencies: `pip install <package-name>`
- Some packages may require specific versions

### Generation Errors
- Try with fewer room types
- Adjust refinement iterations
- Check that at least one room type is selected

## Command Line Usage

You can also use the architectural inference directly:

```python
from inference import ArchitecturalHouseGANInference

generator = ArchitecturalHouseGANInference()
room_types = ['living_room', 'bedroom', 'kitchen', 'bathroom']
floorplan_image, masks, room_ids, real_nodes = generator.generate_floorplan(room_types)
floorplan_image.save('my_floorplan.png')
```

## Contributing

To extend the interface:
- Add new room types to `ROOM_CLASS` dictionary
- Implement more sophisticated graph connectivity
- Add manual room constraint controls
- Improve visualization options