# Agents & Modules

The Agadmator Augmentor pipeline is conceptually divided into several autonomous "Agents" or modules. While currently implemented as linear functional pipelines, the architecture is designed to support independent, asynchronous agents in the future.

## 1. The Extraction Agent (Computer Vision & DSP)
Responsible for observing the raw video and audio feeds and translating them into structured sensory data.
- **Audio Sub-module:** Listens for volume spikes and outputs a timeline of "events".
- **Spatial Sub-module:** Continuously scans the localized board region for chess.com UI elements (Yellow Squares, Red Squares, Yellow Arrows).
- **Role:** It does not know the rules of chess; it simply reports *what* changed and *when*, acting as the eyes and ears of the system.

## 2. The Verification Agent (Chess Engine Logic)
Acts as the game logic authority.
- Receives the sensory reports (e.g., "Squares b1 and b7 lit up at 45.2 seconds") from the Extraction Agent.
- Maintains an internal `python-chess` board state.
- Validates the sensory data against standard chess physics. If the Extraction Agent thinks a Pawn moved horizontally, the Verification Agent rejects it.
- **Role:** Filters out sensory hallucinations and false positives, ensuring the output PGN is 100% legal and accurate.

## 3. The Commentary Agent (Future)
Designed to contextualize the human element of the video.
- Collects "non-game" data from the Extraction Agent (Red Squares, Yellow Arrows).
- Transcribes the audio feed using an LLM or Speech-to-Text API.
- **Role:** Correlates the streamer's drawings with their spoken words. (e.g., Streamer draws an arrow from g2 to b7 and says "The bishop is eyeing the queen"; the Commentary Agent logs this insight alongside the timeline).

## 4. The Augmentation Agent (Future)
The final orchestrator that builds the end-user experience.
- Takes the verified PGN, timestamps, and Commentary Agent metadata.
- Interfaces with an external engine (like Stockfish) to evaluate the engine's opinion vs. the human's played move.
- **Role:** Generates the final output JSON and overlays (e.g., rendering evaluation bars or dynamic arrows back onto the video).