# Usage Guide

This document explains how to run the move extraction script.

## Prerequisites

- Python 3.6+
- OpenCV for Python
- NumPy

## Installation

You can install the required Python libraries using pip:

```bash
pip install opencv-python numpy
```

## Running the Analysis

The primary script for analysis is `phase1_extract_moves.py`. You can run it from your terminal.

```bash
# Basic usage
python phase1_extract_moves.py "path/to/your/video.mp4"

# To limit analysis to the first 5 minutes (300 seconds)
python phase1_extract_moves.py "path/to/your/video.mp4" --max-duration 300
```

### Configuration

Several parameters can be adjusted directly in `phase1_extract_moves.py` to fine-tune detection for different video layouts or styles:

- `BOARD_X`, `BOARD_Y`: The top-left corner coordinates of the chessboard.
- `SQUARE_SIZE`: The size of a single square in pixels.
- `CHANGE_THRESHOLD`: The sensitivity for detecting a change on a square.
- `INTRO_SKIP`: Seconds of video to skip at the beginning.
- `SAVE_DEBUG_SCREENSHOTS`: Set to `True` to save images of detected moves in the `output` directory.