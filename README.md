# Multimodal Proctoring Project – Data Collector

This repository contains the data collection module for a multimodal proctoring system. It uses computer vision and sensor data to monitor test-takers through face, hand, head pose, and interaction tracking.

## Features
- Real-time face landmark detection using MediaPipe
- Hand landmark detection and tracking
- Head pose estimation
- GUI-based data collector with event logging
- Webcam and window interaction tests

## Repository Structure
```plain
├── Dataset_Raw/               # Raw collected data (ignored by git)
├── Dataset_Structured/         # Processed dataset (ignored by git)
├── data_collector.py           # Main data collection script
├── event_logger.py             # Logging utilities
├── face_landmarker.task        # MediaPipe face landmark model
├── face_mesh_test.py           # Test script for face mesh
├── gui_collector.py            # GUI for data collection
├── hand_landmarker.task        # MediaPipe hand landmark model
├── head_pose_test.py           # Head pose estimation test
├── interaction_test.py         # Test for user interactions
├── webcam_test.py              # Webcam functionality test
├── window_test.py              # Window management test
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files/folders ignored by git
└── README.md                   # This file
```

## Prerequisites
- Python 3.8 or higher
- Git
- A webcam (for testing)

## How to Clone and Run on Another Computer

Follow these steps to set up the project on a new machine:

### 1. Clone the Repository
Open a terminal (Command Prompt, PowerShell, or Git Bash) and run:
```
git clone https://github.com/Laam24/Multimodal-Proctoring-Project-data-collector.git
cd Multimodal-Proctoring-Project-data-collector
```

### 2. Create and Activate a Virtual Environment (Recommended)
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run a Test Script
Verify everything works by running a simple test:
```bash
python webcam_test.py
```
This should open your webcam and display the feed.

### 5. Start the GUI Data Collector
```bash
python gui_collector.py
```
Follow the on-screen instructions to collect multimodal data.

## Important Notes
- The `.task` files (face/hand landmark models) are included in the repository. No need to download them separately.
- The `Dataset_Raw` and `Dataset_Structured` folders are ignored by Git (see `.gitignore`). If you need the actual datasets, obtain them from the original source or generate them by running the data collector.
- Large files committed before adding `.gitignore` are still tracked in the repository history. If you want to remove them permanently, please contact the repository maintainer.

## Troubleshooting
- If you encounter `ModuleNotFoundError`, ensure your virtual environment is activated and all dependencies are installed.
- If the webcam doesn't open, check your camera permissions and try another USB port.
- For MediaPipe issues, refer to the [official documentation](https://google.github.io/mediapipe/).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
```
