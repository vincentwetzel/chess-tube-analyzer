# Agadmator Augmentor

A project to analyze chess videos from sources like Agadmator's Chess Channel and extract game data, including moves, positions, and timestamps.

## Overview

This toolset provides utilities to process chess video files, identify board states, and reconstruct the game played.

## Documentation

- [Architecture & System Design](architecture.md)
- [Agents & Modules](agents.md)

### Phase 1: Move Extraction

The initial phase focuses on extracting chess moves directly from the video. The current implementation uses a hybrid audio-visual approach for high accuracy, factoring in specific chess.com UI mechanics:

**UI Mechanics & Behaviors:**
- **Audio Triggers:** Plies (half moves) trigger precisely when the player releases their mouse. This plays a sound and initiates a snap animation to center the piece. Visual snapshots must be slightly delayed after the sound to ensure the piece has settled.
- **Highlighting:** The UI highlights the *previous* move's origin and destination squares in light yellow.
  - **2 Yellow Squares:** Indicates a completed move (origin square is empty, destination square has the piece).
  - **3 Yellow Squares:** Indicates a completed move + the user has currently picked up another piece (which highlights its origin square as well).
- **Red Squares:** Agadmator occasionally right-clicks to highlight squares in red to draw attention to them. They do not alter game logic, but the system actively tracks these semi-transparent overlays (even under pieces) for UI context.
- **Yellow Arrows:** Drawn by the streamer to point out specific dynamics or future moves on the board. These do not count as plies but are tracked to monitor what the streamer is emphasizing.

1.  **Audio Peak Detection:** Analyzes the video's audio track to find volume spikes corresponding to the mouse-release sounds, giving us the exact timestamps of every ply.
2.  **Board Localization:** Uses multi-pass template matching with reference assets (`board.png`) to identify the sub-pixel exact coordinates of the chess board.
3.  **Visual Extraction:** By delaying slightly after the audio peak (to allow the piece to snap to center) or by utilizing the UI's yellow highlighted squares, it determines the origin and destination of the move.
4.  **Legal Move Verification:** It generates all legal moves for the current position using a chess engine, evaluates them against the visually altered squares, and logs the highest-scoring legal match.

## Getting Started

To get started with running the analysis, please see the usage guide.

- **Usage Guide**

This project is under development. Future phases may include template matching using piece assets, OCR for on-screen text, and integration with chess engines for deeper analysis.