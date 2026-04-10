# Agadmator Augmentor

A project to analyze chess videos from sources like Agadmator's Chess Channel and extract game data, including moves, positions, and timestamps.

## Overview

This toolset provides utilities to process chess video files, identify board states, and reconstruct the game played. The pipeline is purely visual — it uses the chess.com UI itself (highlights, clocks, arrows) as an absolute state machine to achieve high accuracy without relying on audio or external metadata.

## Documentation

- [Usage Guide](docs/USAGE.md) — Installation, CLI, and output format
- [Architecture & System Design](architecture.md) — How the visual pipeline works
- [Agents & Modules](agents.md) — Conceptual agent breakdown
- [Project Plan](PROJECT_PLAN.md) — Roadmap and current status

## Current Status: Full Pipeline Working

All 4 phases are implemented and tested:

### ✅ Phase 1: Move Extraction (`video_extractor.py`)
- **Board Localization:** Multi-pass template matching with sub-pixel scale sweeping
- **Visual State Machine:** Frames polled at 5 FPS; compared against the last settled board snapshot
- **UI Validation:** 4-layer filter — yellow square highlights, white hover-box rejection, clock turn verification, analysis revert detection
- **Engine Verification:** `python-chess` validates strictly legal moves
- **Clock OCR:** Tesseract extracts active player and remaining time
- **Output:** `output/analysis.json` with moves, timestamps, FENs, and clocks

### ✅ Phase 2: Stockfish Analysis (`stockfish_analysis.py`)
- Auto-detects Stockfish at `C:\stockfish-windows-x86-64-avx2\stockfish\`
- Analyzes all detected positions with top 3 moves and evaluations
- Output: `output/analysis_analyzed.json`

### ✅ Phase 3: Overlay Renderer (`overlay_renderer.py`)
- Generates visual overlay with board, eval bar, arrows, PV text
- Tested with Stockfish data

### ✅ Phase 4: Video Composition (`video_compositor.py`)
- Composites overlay onto original video
- Output: `output/output_with_analysis.mp4`

## Quick Start

```bash
# Install dependencies
pip install opencv-python numpy pytesseract python-chess moviepy tqdm

# Run move extraction on a video
python phase1_extract_moves.py "path/to/video.mp4"

# Output: output/analysis.json
```

See [USAGE.md](docs/USAGE.md) for full details.

## How It Works

The extractor treats the chess.com UI as a deterministic state machine:

1. **Yellow highlights** mark the origin and destination of every completed move
2. **White hover boxes** appear when a piece is dragged — these frames are rejected
3. **Clock turns** must match the expected active player after a move
4. **Red squares** and **yellow arrows** are tracked for streamer commentary context (not counted as plies)
5. **Board reverts** are detected by matching grayscale captures against an internal history

This multi-layer validation ensures only fully settled, legal board states are recorded.

## Project Structure

```
AgadmatorAugmentor/
├── phase1_extract_moves.py      # CLI entry point for move extraction
├── video_extractor.py           # Core ChessVideoExtractor class
├── test_video_extractor.py      # Unit tests for all visual detectors
├── assets/
│   ├── board/board.png          # Required: pristine board image for localization
│   ├── board/red_board.png      # Optional: board with red highlights for threshold calibration
│   ├── sample_games_*/          # Test videos with ground-truth PGNs
│   └── ...
├── debug_screenshots/           # Auto-generated debug output
├── output/                      # Generated JSON and video files
├── docs/
│   └── USAGE.md
├── README.md
├── architecture.md
├── agents.md
└── PROJECT_PLAN.md
```

## Known Issues

- Detection rate is ~80% for the first 10 moves in a 3-minute test
- ~10% of positions detected as "? (...)" when departure/arrival squares are unclear
- Video encoding in Python is slow (16 min video takes ~10 min to process)
- Some moves are missed due to subtle board changes or hand occlusion
