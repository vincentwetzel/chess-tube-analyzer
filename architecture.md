# Architecture & System Design

The Agadmator Augmentor uses a hybrid audio-visual processing pipeline to analyze chess videos. Rather than relying solely on visual state-diffing, the architecture isolates specific UI behaviors from chess.com to achieve extremely high accuracy.

## 1. Audio Peak Detection
To pinpoint the exact timestamp of a chess ply, the system analyzes the video's audio track.
- **Energy Envelope:** Converts the audio to mono and calculates a rolling moving average of squared amplitudes (energy).
- **Thresholding:** Establishes a dynamic baseline (`mean + 1.2 * std`) to detect the "clack" sound of a mouse release/piece placement.
- **Cooldown:** Enforces a 0.5-second cooldown to avoid double-counting reverb or snap animations.

## 2. Board Localization
To ensure the system works across varying video resolutions, it locates the board dynamically.
- **Multi-Pass Template Matching:** Uses `OpenCV` to compare the first video frame against `board.png`.
- **Scale Sweeping:** Sweeps from 0.3x to 1.5x scale in three passes (Coarse -> Fine -> Exact) to find the sub-pixel perfect coordinates and bounding box of the 8x8 grid.

## 3. UI Element Extraction
The core of the visual pipeline relies on extracting semi-transparent UI elements without being confused by the chess pieces covering them or the wood grain beneath them.

### Yellow Squares (Previous Move)
- **Mathematical Yellowness:** Evaluates pixels using `(R + G) / 2.0 - B`. This isolates the semi-transparent yellow highlight from natural wood colors.
- **Corner Sampling:** Only the outer 12% corners of each square are tested to bypass any chess pieces resting in the center.
- **Edge Detection (Piece vs. Empty):** To determine the direction of the move (from vs. to), Canny edge detection is run on the highlighted squares. The square with more edges contains the piece (Destination).

### Red Squares (Streamer Emphasis)
- **Mathematical Redness:** Evaluates pixels using `R - (G + B) / 2.0`. 
- **Dynamic Thresholding:** Compares the redness of a baseline board against a reference `red_board.png` to find the perfect midpoint threshold, ignoring naturally red wood tones.
- **3-of-4 Rule:** A square is flagged as red if at least 3 of its 4 corners pass the redness threshold, tolerating UI conflicts or partial overlaps.

### Yellow Arrows (Streamer Commentary)
- **HSV Masking:** Isolates highly saturated yellow/orange pixels (Saturation > 165).
- **Ray-Casting:** Instead of brittle contour tracing, the system casts mathematical rays from the center of every active square to every other active square.
- **Overlap & Endpoint Validation:** If a ray is mostly covered by the arrow mask, and both its start and end squares are heavily anchored in yellow, it is considered a valid arrow.
- **Thick Suppression Zones:** Long arrows are processed first. Accepted arrows cast a massive "suppression shadow" over their path to prevent the system from hallucinating small, adjacent sub-branches due to the messy, hand-drawn nature of the arrows.

### Piece Counting
- **Inner Square Cropping:** Crops to the inner 70% of each square.
- **Canny Edges:** Uses a Gaussian blur followed by Canny edge detection to separate piece outlines from flat wood grain.

## 4. Legal Move Verification
Because audio thresholds can occasionally trigger on background noise (e.g., mouse clicks, loud speaking), the visual data acts as a filter.
- The system tracks the exact game state using `python-chess`.
- When an audio peak triggers, the visual diffs of all legal moves are evaluated. 
- If the visual changes are insignificant (below the visual threshold), the audio spike is flagged as a false positive and ignored.

## Future Integrations
- **Piece Classification:** Determining specific piece types (Knight, Bishop, etc.) using contour matching or color profiling.
- **Transcripts & Context:** Aligning the detected Red Squares and Yellow Arrows with speech-to-text outputs to contextualize what the streamer is talking about before a move happens.