import argparse
import json
import os
from video_extractor import ChessVideoExtractor

def main():
    parser = argparse.ArgumentParser(description="Extract chess moves from a video file.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--board-asset", default="assets/board/board.png", help="Path to board template")
    parser.add_argument("--output", default="output/analysis.json", help="Path to save the extracted JSON data")
    
    args = parser.parse_args()
    
    print(f"Starting Phase 1 Extraction on: {args.video_path}")
    
    extractor = ChessVideoExtractor(args.board_asset)
    
    # We don't have a Ground Truth PGN for real un-tested runs, so we pass None
    extracted_moves, _, game_data = extractor.extract_moves_from_video(args.video_path)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(game_data, f, indent=4)
        
    print(f"\nExtraction complete! {len(extracted_moves)} moves extracted.")
    print(f"Game data saved to {args.output}. Ready for Phase 2 (Stockfish)!")

if __name__ == "__main__":
    main()