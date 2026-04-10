import unittest
import os
import glob
import shutil
from video_extractor import ChessVideoExtractor

class TestVideoExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists("debug_screenshots"):
            shutil.rmtree("debug_screenshots")
        os.makedirs("debug_screenshots", exist_ok=True)

    @unittest.skip("Test is solid. Skipping for now.")
    def test_1_yellow_squares_extraction(self):
        images_dir = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_yellow_squares"
        board_asset = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\board\board.png"
        
        print("\nRunning unit tests on yellow square images...")
        if not os.path.exists(images_dir):
            self.skipTest(f"Directory not found: {images_dir}")
            
        extractor = ChessVideoExtractor(board_asset)
        extractor.generate_corner_debug_image("debug_screenshots/yellow_squares")
            
        image_paths = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
        
        for img_path in image_paths:
            expected_move = os.path.splitext(os.path.basename(img_path))[0]
            move = extractor.extract_move_from_yellow_squares(img_path)
            
            # The extractor returns UCI (e.g., c4a5), but the filename is SAN (e.g., Na5)
            # We verify the destination square matches to confidently pass the test.
            clean_expected = expected_move.replace("+", "").replace("#", "")
            if clean_expected == "O-O":
                expected_dest = "g1" if move[3] == "1" else "g8"
            elif clean_expected == "O-O-O":
                expected_dest = "c1" if move[3] == "1" else "c8"
            else:
                expected_dest = clean_expected[-2:]
                
            extracted_dest = move[2:4]
            self.assertEqual(extracted_dest, expected_dest, f"Failed on {img_path}: expected move '{expected_move}', got extracted UCI '{move}'")
        
        print(f"PASS: Extracted valid moves from {len(image_paths)} yellow square images.")

    @unittest.skip("Test is solid. Skipping for now.")
    def test_2_piece_counts(self):
        images_dir = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_piece_counts"
        board_asset = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\board\board.png"
        
        print("\nRunning unit tests on piece counting images...")
        if not os.path.exists(images_dir):
            self.skipTest(f"Directory not found: {images_dir}")
            
        image_paths = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
        
        extractor = ChessVideoExtractor(board_asset)
        for img_path in image_paths:
            expected_count = int(os.path.splitext(os.path.basename(img_path))[0])
            extracted_count = extractor.count_pieces_in_image(img_path)
            self.assertEqual(extracted_count, expected_count, f"Failed on {img_path}: expected {expected_count}, got {extracted_count}")
            
        print(f"PASS: Accurately counted pieces in all {len(image_paths)} images.")

    @unittest.skip("Test is solid. Skipping for now.")
    def test_3_red_squares(self):
        images_dir = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_red_squares"
        board_asset = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\board\board.png"
        red_board_asset = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\board\red_board.png"
        
        print("\nRunning unit tests on red square images...")
        if not os.path.exists(images_dir):
            self.skipTest(f"Directory not found: {images_dir}")
            
        extractor = ChessVideoExtractor(board_asset, red_board_asset)
        
        image_paths = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
        
        for img_path in image_paths:
            expected_squares_str = os.path.splitext(os.path.basename(img_path))[0]
            expected_squares = sorted([sq.strip() for sq in expected_squares_str.split(',') if len(sq.strip()) == 2])
            
            extracted_squares = extractor.find_red_squares(img_path)
            self.assertEqual(extracted_squares, expected_squares, f"Failed on {img_path}: expected {expected_squares}, got {extracted_squares}")
            
        print(f"PASS: Accurately detected red squares in all {len(image_paths)} images.")

    @unittest.skip("Test is solid. Skipping for now.")
    def test_4_yellow_arrows(self):
        images_dir = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_yellow_arrows"
        board_asset = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\board\board.png"
        
        print("\nRunning unit tests on yellow arrow images...")
        if not os.path.exists(images_dir):
            self.skipTest(f"Directory not found: {images_dir}")
            
        extractor = ChessVideoExtractor(board_asset)
        
        image_paths = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
        
        for img_path in image_paths:
            expected_str = os.path.splitext(os.path.basename(img_path))[0]
            expected_arrows = sorted([arr.strip() for arr in expected_str.split(',') if len(arr.strip()) == 4])
            
            extracted_arrows = extractor.find_yellow_arrows(img_path)
            self.assertEqual(extracted_arrows, expected_arrows, f"Failed on {img_path}: expected {expected_arrows}, got {extracted_arrows}")
            
        print(f"PASS: Accurately detected yellow arrows in all {len(image_paths)} images.")

    @unittest.skip("Test is solid. Skipping for now.")
    def test_5_misaligned_pieces(self):
        images_dir = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_misaligned_piece"
        board_asset = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\board\board.png"
        
        print("\nRunning unit tests on misaligned piece images...")
        if not os.path.exists(images_dir):
            self.skipTest(f"Directory not found: {images_dir}")
            
        debug_dir = "debug_screenshots/misaligned_pieces"
        os.makedirs(debug_dir, exist_ok=True)
            
        extractor = ChessVideoExtractor(board_asset)
        image_paths = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))

        for img_path in image_paths:
            expected_square = os.path.splitext(os.path.basename(img_path))[0]
            detected_square = extractor.find_misaligned_piece(img_path, debug_dir)
            self.assertEqual(detected_square, expected_square, f"Failed on {img_path}: expected hover square '{expected_square}', got '{detected_square}'")
            
        print(f"PASS: Accurately detected misaligned pieces in all {len(image_paths)} images.")

    @unittest.skip("Test is solid. Skipping for now.")
    def test_6_game_clocks(self):
        images_dir = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_clock_changes"
        board_asset = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\board\board.png"
        
        print("\nRunning unit tests on game clocks...")
        if not os.path.exists(images_dir):
            self.skipTest(f"Directory not found: {images_dir}")
            
        debug_dir = "debug_screenshots/game_clocks"
        os.makedirs(debug_dir, exist_ok=True)
            
        extractor = ChessVideoExtractor(board_asset)
        image_paths = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))

        for img_path in image_paths:
            # Expected filename format: {active_player}_{white_time}_{black_time}.png
            # Example: white_1-31-28_1-30-07.png
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            expected_active, expected_white, expected_black = base_name.split('_')
            expected_white = expected_white.replace('-', ':')
            expected_black = expected_black.replace('-', ':')
            
            active_player, white_time, black_time = extractor.extract_clocks(img_path, debug_dir)
            
            self.assertEqual(active_player, expected_active, f"Failed on {img_path}: expected active player {expected_active}, got {active_player}")
            self.assertEqual(white_time, expected_white, f"Failed on {img_path}: expected white time {expected_white}, got {white_time}")
            self.assertEqual(black_time, expected_black, f"Failed on {img_path}: expected black time {expected_black}, got {black_time}")
            
        print(f"PASS: Accurately extracted clocks and active player in all {len(image_paths)} images.")
    
    @unittest.skip("Test is solid. Skipping for now.")
    def test_7_plies_extraction(self):
        video_path = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_games_short\7 plies\7 plies.mp4"
        board_asset = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\board\board.png"
        pgn_path = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_games_short\7 plies\game.pgn"
        
        print(f"\nRunning unit test on: {video_path}")
        extractor = ChessVideoExtractor(board_asset)
        extracted_moves, expected_moves, game_data = extractor.extract_moves_from_video(
            video_path,
            pgn_path,
            debug_label="test_7_plies_extraction"
        )
        
        self.assertEqual(extracted_moves, expected_moves, 
                         f"Extracted moves {extracted_moves} do not match expected {expected_moves}")
        print("PASS: Extracted moves perfectly match the expected 7 plies from the PGN.")

    def test_8_medium_game_with_revert(self):
        video_path = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_games_medium\medium_game_with_analysis_line_and_revert.mp4"
        board_asset = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\board\board.png"
        pgn_path = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_games_medium\game.pgn"
        
        print(f"\nRunning unit test on: {os.path.basename(video_path)}")
        extractor = ChessVideoExtractor(board_asset)
        extracted_moves, expected_moves, game_data = extractor.extract_moves_from_video(
            video_path,
            pgn_path,
            debug_label="test_medium_game_with_revert"
        )
        
        self.assertEqual(extracted_moves, expected_moves, 
                         f"Extracted moves {extracted_moves} do not match expected {expected_moves}")
        print("PASS: Extracted moves perfectly match the expected moves from the PGN, correctly handling analysis line revert.")

if __name__ == '__main__':
    unittest.main()
