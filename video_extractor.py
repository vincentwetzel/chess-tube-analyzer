import cv2
import numpy as np
import chess
import chess.pgn
from moviepy import VideoFileClip
import os
import re
import pytesseract
import shutil
from tqdm import tqdm

class ChessVideoExtractor:
    def __init__(self, board_asset_path, red_board_asset_path=None):
        self.board_template = cv2.imread(board_asset_path)
        if self.board_template is None:
            raise FileNotFoundError(f"Could not load board asset at: {board_asset_path}")
            
        if red_board_asset_path:
            self.red_board_template = cv2.imread(red_board_asset_path)
        else:
            self.red_board_template = None

    def locate_board(self, img_bgr):
        """Performs multi-pass template matching to find the exact board coordinates and scale."""
        best_scale = 1.0
        best_val = -1
        # Pass 1: Coarse search (0.3x to 1.5x)
        for scale in np.linspace(0.3, 1.5, 25):
            rw, rh = int(self.board_template.shape[1] * scale), int(self.board_template.shape[0] * scale)
            if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
            res = cv2.matchTemplate(img_bgr, cv2.resize(self.board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_val: best_val, best_scale = max_val, scale
                
        # Pass 2: Fine search
        best_val = -1
        for scale in np.linspace(best_scale - 0.05, best_scale + 0.05, 21):
            rw, rh = int(self.board_template.shape[1] * scale), int(self.board_template.shape[0] * scale)
            if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
            res = cv2.matchTemplate(img_bgr, cv2.resize(self.board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_val: best_val, best_scale = max_val, scale
                
        # Pass 3: Exact search
        best_val, best_loc, best_shape = -1, (0, 0), self.board_template.shape[:2]
        for scale in np.linspace(best_scale - 0.01, best_scale + 0.01, 21):
            rw, rh = int(self.board_template.shape[1] * scale), int(self.board_template.shape[0] * scale)
            if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
            res = cv2.matchTemplate(img_bgr, cv2.resize(self.board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val: best_val, best_loc, best_shape = max_val, max_loc, (rh, rw)
                
        bx, by = best_loc
        bh, bw = best_shape
        sq_h, sq_w = bh / 8.0, bw / 8.0
        
        return bx, by, bh, bw, sq_h, sq_w

    def _draw_board_grid(self, image, bx, by, sq_h, sq_w, default_color=(0, 255, 0), thickness=2, highlights=None, draw_labels=False):
        """Draws an 8x8 grid on the image, with optional highlighting and labels."""
        highlights = highlights or {}
        for row in range(8):
            for col in range(8):
                sq_name = chr(ord('a') + col) + str(8 - row)
                color = highlights.get(sq_name, default_color)
                
                y1, y2 = int(by + row * sq_h), int(by + (row + 1) * sq_h)
                x1, x2 = int(bx + col * sq_w), int(bx + (col + 1) * sq_w)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                if draw_labels:
                    # Put labels on the grid for clarity
                    cv2.putText(image, sq_name, (x1 + 5, int(y1 + sq_h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    def _sanitize_debug_label(self, label):
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
        return sanitized or "run"

    def extract_moves_from_video(self, video_path, pgn_path=None, debug_label=None):
        expected_moves = []
        if pgn_path:
            print("Loading Ground Truth PGN for test verification...")
            with open(pgn_path, 'r') as pgn_file:
                game = chess.pgn.read_game(pgn_file)
            
            expected_moves = [move.uci() for move in game.mainline_moves()]
            print(f"Expected moves ({len(expected_moves)} plies): {expected_moves}\n")

        clip = VideoFileClip(video_path)

        print("Locating board coordinates using template matching...")
        first_frame = clip.get_frame(0)
        first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        
        print("Performing multi-pass template matching to find exact board scale...")
        bx, by, bh, bw, sq_h, sq_w = self.locate_board(first_frame_bgr)

        debug_label = debug_label or os.path.splitext(os.path.basename(video_path))[0]
        debug_dir = os.path.join("debug_screenshots", "video_extraction", self._sanitize_debug_label(debug_label))
        
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
            
        print("Generating debug screenshot for initial board...")
        os.makedirs(debug_dir, exist_ok=True)
        debug_board_img = first_frame_bgr.copy()
        # Draw the initial board grid with labels for debugging
        self._draw_board_grid(debug_board_img, bx, by, sq_h, sq_w, default_color=(0, 255, 0), thickness=2, draw_labels=True)
        cv2.imwrite(os.path.join(debug_dir, "00_initial_board_0.00s.png"), debug_board_img)

        print("Scanning video frames at fixed intervals to calculate moves...")
        extracted_moves = []
        move_timestamps = []
        board = chess.Board()
        board_history = [board.copy()] # History of board states
        
        def rollback_to_history_index(target_index, timestamp, reason):
            nonlocal board, extracted_moves, board_history, clock_history, board_image_history, move_timestamps
            num_reverted_moves = len(extracted_moves) - target_index
            print(f"ANALYSIS REVERT DETECTED at {timestamp:.2f}s ({reason}): Board state snapped back to ply {target_index}.")
            if num_reverted_moves > 0:
                print(f"  Rolling back {num_reverted_moves} analysis moves: {extracted_moves[target_index:]}")
            board = board_history[target_index].copy()
            extracted_moves = extracted_moves[:target_index]
            move_timestamps = move_timestamps[:target_index]
            board_history = board_history[:target_index+1]
            clock_history = clock_history[:target_index+1]
            board_image_history = board_image_history[:target_index+1]

        margin_h = int(sq_h * 0.15)
        margin_w = int(sq_w * 0.15)
        
        def get_max_square_diff(img_a, img_b):
            diff = cv2.absdiff(img_a, img_b)
            max_sq_diff = 0
            for row in range(8):
                for col in range(8):
                    y1, y2 = int(row * sq_h), int((row + 1) * sq_h)
                    x1, x2 = int(col * sq_w), int((col + 1) * sq_w)
                    sq_diff = np.mean(diff[y1+margin_h:y2-margin_h, x1+margin_w:x2-margin_w])
                    if sq_diff > max_sq_diff:
                        max_sq_diff = sq_diff
            return max_sq_diff
        
        def score_moves_for_board(test_board, square_diffs):
            best_m, best_s = None, -1
            for legal_move in test_board.legal_moves:
                from_sq = chess.square_name(legal_move.from_square)
                to_sq = chess.square_name(legal_move.to_square)
                move_score = square_diffs.get(from_sq, 0) + square_diffs.get(to_sq, 0)
                
                if test_board.is_castling(legal_move):
                    if legal_move.to_square == chess.G1: move_score += square_diffs.get('h1', 0) + square_diffs.get('f1', 0)
                    elif legal_move.to_square == chess.C1: move_score += square_diffs.get('a1', 0) + square_diffs.get('d1', 0)
                    elif legal_move.to_square == chess.G8: move_score += square_diffs.get('h8', 0) + square_diffs.get('f8', 0)
                    elif legal_move.to_square == chess.C8: move_score += square_diffs.get('a8', 0) + square_diffs.get('d8', 0)
                
                if test_board.is_en_passant(legal_move):
                    captured_pawn_sq_idx = chess.square(chess.square_file(legal_move.to_square), chess.square_rank(legal_move.from_square))
                    captured_pawn_sq_name = chess.square_name(captured_pawn_sq_idx)
                    move_score += square_diffs.get(captured_pawn_sq_name, 0)
                    
                if move_score > best_s:
                    best_s = move_score
                    best_m = legal_move
            return best_m, best_s

        # Extract initial clocks
        print("Extracting initial clock states...")
        initial_frame_bgr = cv2.cvtColor(clip.get_frame(0), cv2.COLOR_RGB2BGR)
        init_active, init_white, init_black = self.extract_clocks(initial_frame_bgr)
        clock_history = [{'active': init_active, 'white': init_white, 'black': init_black}]
        
        initial_frame_gray = cv2.cvtColor(initial_frame_bgr, cv2.COLOR_BGR2GRAY)
        board_image_history = [initial_frame_gray[by:by+bh, bx:bx+bw].copy()]
        
        time_step = 0.2 # Poll at 5 FPS

        for t in tqdm(np.arange(0.0, clip.duration, time_step), desc="Scanning video", unit="frames"):
            if board.is_game_over():
                break
                
            frame_after_rgb = clip.get_frame(t)
            frame_after_bgr = cv2.cvtColor(frame_after_rgb, cv2.COLOR_RGB2BGR)
            frame_after_gray = cv2.cvtColor(frame_after_bgr, cv2.COLOR_BGR2GRAY)
            
            board_after = frame_after_gray[by:by+bh, bx:bx+bw]
            board_before = board_image_history[-1]
            
            # Check if the board silently reverted
            if len(board_image_history) > 1:
                current_state_diff = get_max_square_diff(board_after, board_image_history[-1])
                if current_state_diff > 15.0: # The board has changed silently
                    best_match_idx = None
                    best_match_diff = float('inf')
                    # Look backwards through history
                    for idx in range(len(board_image_history) - 2, -1, -1):
                        diff = get_max_square_diff(board_after, board_image_history[idx])
                        if diff < best_match_diff:
                            best_match_diff = diff
                            best_match_idx = idx
                            
                    if best_match_idx is not None and best_match_diff < 15.0:
                        rollback_to_history_index(best_match_idx, t, "board visually matched past state")
                        board_before = board_image_history[-1]
            
            diff = cv2.absdiff(board_after, board_before)
            
            square_diffs = {}
            
            for row in range(8):
                for col in range(8):
                    y1, y2 = int(row * sq_h), int((row + 1) * sq_h)
                    x1, x2 = int(col * sq_w), int((col + 1) * sq_w)
                    sq_name = chr(ord('a') + col) + str(8 - row)
                    square_diffs[sq_name] = np.mean(diff[y1+margin_h:y2-margin_h, x1+margin_w:x2-margin_w])
                    
            if not square_diffs or max(square_diffs.values()) < 15.0:
                continue
                
            best_move, best_score = score_moves_for_board(board, square_diffs)
            
            # Filter out false positive visual changes using a visual change threshold
            inverse_recent_move = False
            if best_move and extracted_moves:
                inverse_uci = chess.square_name(best_move.to_square) + chess.square_name(best_move.from_square)
                inverse_recent_move = inverse_uci in extracted_moves[-4:]

            if best_score > 25.0 and best_move and not (inverse_recent_move and best_score < 70.0):
                board_after_bgr = frame_after_bgr[by:by+bh, bx:bx+bw]
                
                from_sq_name = chess.square_name(best_move.from_square)
                to_sq_name = chess.square_name(best_move.to_square)
                
                # Validation 1 & 2: Yellow Squares
                b, g, r = cv2.split(board_after_bgr.astype(float))
                yellowness_map = (r + g) / 2.0 - b
                ch, cw = int(sq_h * 0.12), int(sq_w * 0.12)
                
                def get_yellowness(sq_name):
                    col, row = ord(sq_name[0]) - ord('a'), 8 - int(sq_name[1])
                    y1, y2 = int(row * sq_h), int((row + 1) * sq_h)
                    x1, x2 = int(col * sq_w), int((col + 1) * sq_w)
                    corner_mask = np.concatenate([
                        yellowness_map[y1:y1+ch, x1:x1+cw].flatten(), 
                        yellowness_map[y1:y1+ch, x2-cw:x2].flatten(),
                        yellowness_map[y2-ch:y2, x1:x1+cw].flatten(), 
                        yellowness_map[y2-ch:y2, x2-cw:x2].flatten()
                    ])
                    return np.mean(corner_mask)

                y_from = get_yellowness(from_sq_name)
                y_to = get_yellowness(to_sq_name)
                
                if y_from < 40.0 or y_to < 40.0:
                    continue
                    
                # Validation 3: Misaligned piece (Hover box)
                lower_white = np.array([160, 160, 160])
                upper_white = np.array([255, 255, 255])
                white_mask = cv2.inRange(board_after_bgr, lower_white, upper_white)
                
                def has_hover_box(sq_name):
                    col, row = ord(sq_name[0]) - ord('a'), 8 - int(sq_name[1])
                    y1, y2 = int(row * sq_h), int((row + 1) * sq_h)
                    x1, x2 = int(col * sq_w), int((col + 1) * sq_w)
                    thickness = max(3, int(sq_w * 0.08))
                    top = white_mask[y1:y1+thickness, x1:x2]
                    bottom = white_mask[y2-thickness:y2, x1:x2]
                    left = white_mask[y1:y2, x1:x1+thickness]
                    right = white_mask[y1:y2, x2-thickness:x2]
                    t_ratio = np.count_nonzero(np.max(top, axis=0)) / max(1, x2 - x1)
                    b_ratio = np.count_nonzero(np.max(bottom, axis=0)) / max(1, x2 - x1)
                    l_ratio = np.count_nonzero(np.max(left, axis=1)) / max(1, y2 - y1)
                    r_ratio = np.count_nonzero(np.max(right, axis=1)) / max(1, y2 - y1)
                    return sum(1 for r in [t_ratio, b_ratio, l_ratio, r_ratio] if r > 0.10) >= 2

                if has_hover_box(to_sq_name):
                    continue

                active_player, white_time, black_time = self.extract_clocks(frame_after_bgr)
                
                # Validation 4: Clock Turn Validation
                # Because we continuously poll, if the UI clock background lags slightly behind the piece animation,
                # we simply reject it here and naturally catch it in the next frame when the clock finally updates!
                if active_player:
                    test_board = board.copy()
                    test_board.push(best_move)
                    expected_active = "white" if test_board.turn == chess.WHITE else "black"
                    
                    if active_player != expected_active:
                        continue

                extracted_moves.append(best_move.uci())
                move_timestamps.append(round(t, 2))
                board.push(best_move)
                board_history.append(board.copy())
                clock_history.append({'active': active_player, 'white': white_time, 'black': black_time})
                board_image_history.append(board_after.copy())
                print(f"Ply {len(extracted_moves)}: visually detected {best_move.uci()} at {t:.2f}s (confidence score: {best_score:.2f})")
                
                # Generate debug screenshot for the move
                for sq_idx in [best_move.from_square, best_move.to_square]:
                    file = chess.square_file(sq_idx)
                    rank = chess.square_rank(sq_idx)
                    row, col = 7 - rank, file
                    y1, y2 = int(by + row * sq_h), int(by + (row + 1) * sq_h)
                    x1, x2 = int(bx + col * sq_w), int(bx + (col + 1) * sq_w)
                    cv2.rectangle(frame_after_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                cv2.imwrite(os.path.join(debug_dir, f"{len(extracted_moves):02d}_before_{t:.2f}s.png"), board_before)
                cv2.imwrite(os.path.join(debug_dir, f"{len(extracted_moves):02d}_after_{t:.2f}s.png"), board_after)
                cv2.imwrite(os.path.join(debug_dir, f"{len(extracted_moves):02d}_diff_{t:.2f}s.png"), diff)
                cv2.imwrite(os.path.join(debug_dir, f"{len(extracted_moves):02d}_{best_move.uci()}_{t:.2f}s.png"), frame_after_bgr)
                
        # Thoroughly close all MoviePy internal FFMPEG subprocess pipes to prevent ResourceWarnings
        if clip is not None:
            if hasattr(clip, 'reader') and hasattr(clip.reader, 'proc') and clip.reader.proc:
                try:
                    if clip.reader.proc.stdout: clip.reader.proc.stdout.close()
                    if clip.reader.proc.stderr: clip.reader.proc.stderr.close()
                except Exception: pass
            clip.close()
            
        game_data = {
            "moves": extracted_moves,
            "timestamps": move_timestamps,
            "fens": [b.fen() for b in board_history],
            "clocks": clock_history
        }
        
        return extracted_moves, expected_moves, game_data

    def generate_corner_debug_image(self, output_dir):
        """Generates an empty board image highlighting the 4 corners being tested for yellowness."""
        print("Generating corner debug regions image...")
        board_img = self.board_template.copy()
            
        bh, bw = board_img.shape[:2]
        sq_h, sq_w = bh / 8.0, bw / 8.0
        ch, cw = int(sq_h * 0.12), int(sq_w * 0.12) # 12% of the square size for tight corners
        
        debug_img = board_img.copy()
        color = (255, 0, 255) # Bright Magenta
        
        for row in range(8):
            for col in range(8):
                y1, y2 = int(row * sq_h), int((row + 1) * sq_h)
                x1, x2 = int(col * sq_w), int((col + 1) * sq_w)
                cv2.rectangle(debug_img, (x1, y1), (x1+cw, y1+ch), color, -1)
                cv2.rectangle(debug_img, (x2-cw, y1), (x2, y1+ch), color, -1)
                cv2.rectangle(debug_img, (x1, y2-ch), (x1+cw, y2), color, -1)
                cv2.rectangle(debug_img, (x2-cw, y2-ch), (x2, y2), color, -1)
                
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "00_corner_debug_regions.png"), debug_img)

    def extract_move_from_yellow_squares(self, image_path):
        """Analyzes a static image for 2 yellow squares to deduce the previous move."""
        print(f"Analyzing yellow squares in: {os.path.basename(image_path)}")
        img_bgr = cv2.imread(image_path)
        
        bx, by, bh, bw, sq_h, sq_w = self.locate_board(img_bgr)
        
        board_img = img_bgr[by:by+bh, bx:bx+bw]
        
        # Calculate 'yellowness' to perfectly isolate semi-transparent highlights.
        # Yellow is created by High Red + High Green, and Low Blue.
        # By subtracting Blue from the average of Red and Green, we suppress neutral/wood colors
        b, g, r = cv2.split(board_img.astype(float))
        yellowness_map = (r + g) / 2.0 - b
        
        yellow_scores = {}
        ch, cw = int(sq_h * 0.12), int(sq_w * 0.12)
        for row in range(8):
            for col in range(8):
                y1, y2 = int(row * sq_h), int((row + 1) * sq_h)
                x1, x2 = int(col * sq_w), int((col + 1) * sq_w)
                
                # Extract only the 4 corners of the square to avoid pieces covering the yellow center
                corner_mask = np.concatenate([yellowness_map[y1:y1+ch, x1:x1+cw].flatten(), yellowness_map[y1:y1+ch, x2-cw:x2].flatten(),
                                              yellowness_map[y2-ch:y2, x1:x1+cw].flatten(), yellowness_map[y2-ch:y2, x2-cw:x2].flatten()])
                
                sq_name = chr(ord('a') + col) + str(8 - row)
                yellow_scores[sq_name] = np.mean(corner_mask)
                
        sorted_squares = sorted(yellow_scores.items(), key=lambda item: item[1], reverse=True)
        sq1_name, sq2_name = sorted_squares[0][0], sorted_squares[1][0]
        
        # Use edge detection to figure out which yellow square has a piece vs which is empty
        board_gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        
        def get_edge_score(sq_name):
            col, row = ord(sq_name[0]) - ord('a'), 8 - int(sq_name[1])
            y1, y2 = int(row * sq_h), int((row + 1) * sq_h)
            x1, x2 = int(col * sq_w), int((col + 1) * sq_w)
            margin_h, margin_w = int(sq_h * 0.15), int(sq_w * 0.15)
            sq_gray = board_gray[y1+margin_h:y2-margin_h, x1+margin_w:x2-margin_w]
            # Blur slightly to eliminate wood grain texture, leaving only piece outlines
            edges = cv2.Canny(cv2.GaussianBlur(sq_gray, (3, 3), 0), 50, 150)
            return np.mean(edges)
            
        score1, score2 = get_edge_score(sq1_name), get_edge_score(sq2_name)
        
        # The square with more edges has the piece (Destination)
        if score1 > score2:
            to_sq, from_sq = sq1_name, sq2_name
        else:
            to_sq, from_sq = sq2_name, sq1_name
            
        move_uci = from_sq + to_sq
        
        # Save debug image
        os.makedirs("debug_screenshots/yellow_squares", exist_ok=True)
        debug_img = img_bgr.copy()
        for sq, color, label in [(from_sq, (0, 0, 255), "FROM (Empty)"), (to_sq, (0, 255, 0), "TO (Piece)")]:
            col, row = ord(sq[0]) - ord('a'), 8 - int(sq[1])
            cv2.rectangle(debug_img, (int(bx + col * sq_w), int(by + row * sq_h)), 
                          (int(bx + (col + 1) * sq_w), int(by + (row + 1) * sq_h)), color, 3)
            cv2.putText(debug_img, label, (int(bx + col * sq_w) + 5, int(by + row * sq_h) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        cv2.imwrite(f"debug_screenshots/yellow_squares/{os.path.basename(image_path)}_detected_{move_uci}.png", debug_img)
        print(f"  -> Found move: {move_uci}")
        return move_uci

    def count_pieces_in_image(self, image_path):
        """Analyzes a static image and counts the number of chess pieces on the board."""
        print(f"Counting pieces in: {os.path.basename(image_path)}")
        img_bgr = cv2.imread(image_path)
        
        bx, by, bh, bw, sq_h, sq_w = self.locate_board(img_bgr)
        
        board_img = img_bgr[by:by+bh, bx:bx+bw]
        board_gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        
        count = 0
        os.makedirs("debug_screenshots/piece_counts", exist_ok=True)
        debug_img = img_bgr.copy()
        
        for row in range(8):
            for col in range(8):
                y1, y2 = int(row * sq_h), int((row + 1) * sq_h)
                x1, x2 = int(col * sq_w), int((col + 1) * sq_w)
                margin_h, margin_w = int(sq_h * 0.15), int(sq_w * 0.15)
                sq_gray = board_gray[y1+margin_h:y2-margin_h, x1+margin_w:x2-margin_w]
                
                edges = cv2.Canny(cv2.GaussianBlur(sq_gray, (3, 3), 0), 40, 100)
                if np.mean(edges) > 10.0: # Threshold for piece edges vs plain wood
                    count += 1
                    cv2.rectangle(debug_img, (int(bx + x1), int(by + y1)), (int(bx + x2), int(by + y2)), (255, 0, 0), 3)
                    
        cv2.imwrite(f"debug_screenshots/piece_counts/{os.path.basename(image_path)}_counted_{count}.png", debug_img)
        print(f"  -> Found {count} pieces")
        return count

    def find_red_squares(self, image_path):
        """Analyzes a static image to find any squares highlighted in red."""
        print(f"Analyzing red squares in: {os.path.basename(image_path)}")
        img_bgr = cv2.imread(image_path)
        
        bx, by, bh, bw, sq_h, sq_w = self.locate_board(img_bgr)
        
        board_img = img_bgr[by:by+bh, bx:bx+bw]
        
        # Calculate 'redness' map by subtracting average of Green/Blue from Red
        b, g, r = cv2.split(board_img.astype(float))
        redness_map = r - (g + b) / 2.0
        
        # Determine dynamic threshold using the normal board and red_board.png
        tb, tg, tr = cv2.split(self.board_template.astype(float))
        normal_redness = np.mean(tr - (tg + tb) / 2.0)
        threshold = normal_redness + 35.0  # Safe fallback
        
        if self.red_board_template is not None:
            # In case red_board.png is a full screen image, find the board within it
            res = cv2.matchTemplate(self.red_board_template, self.board_template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            rx, ry = max_loc
            rh, rw = self.board_template.shape[:2]
            
            if ry+rh <= self.red_board_template.shape[0] and rx+rw <= self.red_board_template.shape[1]:
                red_board_cropped = self.red_board_template[ry:ry+rh, rx:rx+rw]
                rb, rg, rr = cv2.split(red_board_cropped.astype(float))
                red_redness = np.mean(rr - (rg + rb) / 2.0)
                # The midpoint between normal wood and red overlay
                threshold = (normal_redness + red_redness) / 2.0
            
        red_squares = []
        ch, cw = int(sq_h * 0.12), int(sq_w * 0.12)
        
        for row in range(8):
            for col in range(8):
                y1, y2 = int(row * sq_h), int((row + 1) * sq_h)
                x1, x2 = int(col * sq_w), int((col + 1) * sq_w)
                
                corners = [
                    redness_map[y1:y1+ch, x1:x1+cw], redness_map[y1:y1+ch, x2-cw:x2],
                    redness_map[y2-ch:y2, x1:x1+cw], redness_map[y2-ch:y2, x2-cw:x2]
                ]
                
                # Pass if 3 out of 4 corners are heavily tinted red
                red_corners_count = sum(1 for corner in corners if np.mean(corner) > threshold)
                if red_corners_count >= 3:
                    sq_name = chr(ord('a') + col) + str(8 - row)
                    red_squares.append(sq_name)
                    
        red_squares.sort()
        
        os.makedirs("debug_screenshots/red_squares", exist_ok=True)
        debug_img = img_bgr.copy()
        for sq in red_squares:
            col, row = ord(sq[0]) - ord('a'), 8 - int(sq[1])
            cv2.rectangle(debug_img, (int(bx + col * sq_w), int(by + row * sq_h)), 
                          (int(bx + (col + 1) * sq_w), int(by + (row + 1) * sq_h)), (0, 0, 255), 3)
            
        cv2.imwrite(f"debug_screenshots/red_squares/{os.path.basename(image_path)}_detected.png", debug_img)
        print(f"  -> Found red squares: {','.join(red_squares) if red_squares else 'None'}")
        
        return red_squares

    def find_yellow_arrows(self, image_path):
        """Analyzes a static image to find yellow arrows drawn on the board."""
        print(f"Analyzing yellow arrows in: {os.path.basename(image_path)}")
        img_bgr = cv2.imread(image_path)
        
        bx, by, bh, bw, sq_h, sq_w = self.locate_board(img_bgr)
        
        board_img = img_bgr[by:by+bh, bx:bx+bw]
        
        # Convert to HSV to find highly saturated yellow/orange arrows
        hsv = cv2.cvtColor(board_img, cv2.COLOR_BGR2HSV)
        # Tighten saturation strictly to 150+ to completely ignore muddy yellow squares
        lower_arrow = np.array([10, 165, 165])
        upper_arrow = np.array([40, 255, 255])
        arrow_mask = cv2.inRange(hsv, lower_arrow, upper_arrow)
        
        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        arrow_mask = cv2.morphologyEx(arrow_mask, cv2.MORPH_OPEN, kernel)
        
        sq_centers = {}
        for r in range(8):
            for c in range(8):
                sq_name = chr(ord('a') + c) + str(8 - r)
                cx, cy = int((c + 0.5) * sq_w), int((r + 0.5) * sq_h)
                sq_centers[sq_name] = (cx, cy)
                
        # Pre-filter squares to only check those that have some arrow mask passing through them
        active_squares = {}
        for sq in sq_centers:
            c, r = ord(sq[0]) - ord('a'), 8 - int(sq[1])
            x1, y1 = int(c * sq_w), int(r * sq_h)
            x2, y2 = int((c + 1) * sq_w), int((r + 1) * sq_h)
            active_squares[sq] = cv2.countNonZero(arrow_mask[y1:y2, x1:x2]) > (sq_w * sq_h * 0.015)
            
        potential_lines = []
        for sq1 in sq_centers:
            for sq2 in sq_centers:
                if sq1 >= sq2: continue # Compare each pair only once
                if not active_squares[sq1] or not active_squares[sq2]: continue
                
                cx1, cy1 = sq_centers[sq1]
                cx2, cy2 = sq_centers[sq2]
                
                dx, dy = cx2 - cx1, cy2 - cy1
                dist = np.hypot(dx, dy)
                if dist < sq_w * 0.8: continue
                
                line_mask = np.zeros_like(arrow_mask)
                cv2.line(line_mask, (cx1, cy1), (cx2, cy2), 255, int(sq_w * 0.15))
                
                line_area = cv2.countNonZero(line_mask)
                if line_area == 0: continue
                
                overlap = cv2.countNonZero(cv2.bitwise_and(arrow_mask, line_mask))
                if overlap < 0.60 * line_area: # Require majority of line to be yellow
                    continue
                    
                # Endpoint square check to prevent lines from extending past the true arrow
                def check_sq_overlap(sq):
                    c, r = ord(sq[0]) - ord('a'), 8 - int(sq[1])
                    x1, y1 = int(c * sq_w), int(r * sq_h)
                    x2, y2 = int((c + 1) * sq_w), int((r + 1) * sq_h)
                    sq_line = line_mask[y1:y2, x1:x2]
                    sq_area = cv2.countNonZero(sq_line)
                    if sq_area == 0: return 0
                    sq_arrow = arrow_mask[y1:y2, x1:x2]
                    return cv2.countNonZero(cv2.bitwise_and(sq_line, sq_arrow)) / sq_area
    
                # Ensure both endpoints are heavily anchored in yellow pixels to reject spill-over
                if check_sq_overlap(sq1) < 0.35 or check_sq_overlap(sq2) < 0.35:
                    continue
                    
                # Gap checker: Ensure no continuous gap larger than ~1.2 squares (allows piece occlusions but prevents jumping empty squares)
                num_samples = int(dist / (sq_w * 0.20)) # 5 samples per square width
                max_gaps, curr_gaps = 0, 0
                for i in range(num_samples + 1):
                    px = int(cx1 + dx * i / num_samples)
                    py = int(cy1 + dy * i / num_samples)
                    r_patch = int(sq_w * 0.1)
                    y_min, y_max = max(0, py - r_patch), min(arrow_mask.shape[0], py + r_patch)
                    x_min, x_max = max(0, px - r_patch), min(arrow_mask.shape[1], px + r_patch)
                    
                    if cv2.countNonZero(arrow_mask[y_min:y_max, x_min:x_max]) == 0:
                        curr_gaps += 1
                        max_gaps = max(max_gaps, curr_gaps)
                    else:
                        curr_gaps = 0
                        
                if max_gaps > 6: # 6 samples * 0.20 = 1.2 squares max gap
                    continue
                    
                potential_lines.append((sq1, sq2, line_area, line_mask))
                    
        # Sort by length to process the longest continuous lines first (absorbing sub-segments)
        def sq_dist(item):
            return (sq_centers[item[0]][0] - sq_centers[item[1]][0])**2 + (sq_centers[item[0]][1] - sq_centers[item[1]][1])**2
            
        potential_lines.sort(key=sq_dist, reverse=True)
        
        accepted_lines = []
        covered_mask = np.zeros_like(arrow_mask)
        
        for sq1, sq2, line_area, line_mask in potential_lines:
            already_covered = cv2.countNonZero(cv2.bitwise_and(line_mask, covered_mask))
            
            # Accept line if it isn't already handled by a longer overlapping line
            if already_covered < 0.45 * line_area:
                accepted_lines.append((sq1, sq2))
                # Draw a thicker line into covered_mask to aggressively suppress nearby phantom branches
                thick_mask = np.zeros_like(arrow_mask)
                cv2.line(thick_mask, sq_centers[sq1], sq_centers[sq2], 255, int(sq_w * 1.8))
                cv2.bitwise_or(covered_mask, thick_mask, covered_mask)
                
        def step_square(sq, file_step, rank_step):
            file_idx = ord(sq[0]) - ord('a') + file_step
            rank_idx = int(sq[1]) + rank_step
            if 0 <= file_idx < 8 and 1 <= rank_idx <= 8:
                return chr(ord('a') + file_idx) + str(rank_idx)
            return None
    
        def maybe_extend_line(start_sq, end_sq):
            file_delta = (ord(end_sq[0]) - ord(start_sq[0]))
            rank_delta = int(end_sq[1]) - int(start_sq[1])
            file_step = 0 if file_delta == 0 else int(np.sign(file_delta))
            rank_step = 0 if rank_delta == 0 else int(np.sign(rank_delta))
            extended_start, extended_end = start_sq, end_sq
    
            while True:
                candidate = step_square(extended_start, -file_step, -rank_step)
                if candidate is None or not active_squares.get(candidate, False):
                    break
    
                test_mask = np.zeros_like(arrow_mask)
                cv2.line(test_mask, sq_centers[candidate], sq_centers[extended_end], 255, int(sq_w * 0.15))
                test_area = cv2.countNonZero(test_mask)
                if test_area == 0:
                    break
    
                test_overlap = cv2.countNonZero(cv2.bitwise_and(arrow_mask, test_mask)) / test_area
                if test_overlap < 0.70:
                    break
    
                extended_start = candidate
    
            while True:
                candidate = step_square(extended_end, file_step, rank_step)
                if candidate is None or not active_squares.get(candidate, False):
                    break
    
                test_mask = np.zeros_like(arrow_mask)
                cv2.line(test_mask, sq_centers[extended_start], sq_centers[candidate], 255, int(sq_w * 0.15))
                test_area = cv2.countNonZero(test_mask)
                if test_area == 0:
                    break
    
                test_overlap = cv2.countNonZero(cv2.bitwise_and(arrow_mask, test_mask)) / test_area
                if test_overlap < 0.70:
                    break
    
                extended_end = candidate
    
            return extended_start, extended_end
    
        arrows = []
        for sq1, sq2 in accepted_lines:
            sq1, sq2 = maybe_extend_line(sq1, sq2)
            mask1, mask2 = np.zeros_like(arrow_mask), np.zeros_like(arrow_mask)
            cv2.circle(mask1, sq_centers[sq1], int(sq_w * 0.45), 255, -1)
            cv2.circle(mask2, sq_centers[sq2], int(sq_w * 0.45), 255, -1)
            
            # The arrowhead triangle will have significantly more pixel mass than the tail
            count1 = cv2.countNonZero(cv2.bitwise_and(arrow_mask, mask1))
            count2 = cv2.countNonZero(cv2.bitwise_and(arrow_mask, mask2))
            
            arrows.append(f"{sq2}{sq1}" if count1 > count2 else f"{sq1}{sq2}")
            
        arrows = sorted(list(set(arrows)))
        
        os.makedirs("debug_screenshots/yellow_arrows", exist_ok=True)
        debug_img = img_bgr.copy()
        for arr in arrows:
            start, end = arr[:2], arr[2:]
            c1, r1 = ord(start[0]) - ord('a'), 8 - int(start[1])
            c2, r2 = ord(end[0]) - ord('a'), 8 - int(end[1])
            x1, y1 = int(bx + (c1 + 0.5) * sq_w), int(by + (r1 + 0.5) * sq_h)
            x2, y2 = int(bx + (c2 + 0.5) * sq_w), int(by + (r2 + 0.5) * sq_h)
            cv2.arrowedLine(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 4, tipLength=0.3)
            cv2.putText(debug_img, arr, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        cv2.imwrite(f"debug_screenshots/yellow_arrows/{os.path.basename(image_path)}_detected.png", debug_img)
        print(f"  -> Found arrows: {','.join(arrows) if arrows else 'None'}")
        
        return arrows

    def find_misaligned_piece(self, image_path, debug_dir=None):
        """
        Analyzes a static image to find the square a piece is currently being hovered over.
        The UI draws a distinct white box (outline) around the edges of the destination square.
        """
        print(f"Detecting hover square in: {os.path.basename(image_path)}")
        img_bgr = cv2.imread(image_path)
        
        bx, by, bh, bw, sq_h, sq_w = self.locate_board(img_bgr)
        board_img = img_bgr[by:by+bh, bx:bx+bw]
        
        # Isolate white/bright pixels (the hover outline)
        lower_white = np.array([160, 160, 160])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(board_img, lower_white, upper_white)
        
        best_square = None
        max_box_score = -1
        
        thickness = max(3, int(sq_w * 0.08)) # ~8% thickness for the outline
        
        for r in range(8):
            for c in range(8):
                y1, y2 = int(r * sq_h), int((r + 1) * sq_h)
                x1, x2 = int(c * sq_w), int((c + 1) * sq_w)
                
                # Extract the 4 border regions of the square
                top = white_mask[y1:y1+thickness, x1:x2]
                bottom = white_mask[y2-thickness:y2, x1:x2]
                left = white_mask[y1:y2, x1:x1+thickness]
                right = white_mask[y1:y2, x2-thickness:x2]
                
                t_ratio = np.count_nonzero(np.max(top, axis=0)) / max(1, x2 - x1)
                b_ratio = np.count_nonzero(np.max(bottom, axis=0)) / max(1, x2 - x1)
                l_ratio = np.count_nonzero(np.max(left, axis=1)) / max(1, y2 - y1)
                r_ratio = np.count_nonzero(np.max(right, axis=1)) / max(1, y2 - y1)
                
                # As the piece is dragged, it might occlude parts of the box.
                # We ensure at least 2 edges are partially visible.
                ratios = [t_ratio, b_ratio, l_ratio, r_ratio]
                visible_edges = sum(1 for r in ratios if r > 0.10)
                
                if visible_edges >= 2:
                    score = sum(ratios)
                    if score > max_box_score:
                        max_box_score = score
                        best_square = chr(ord('a') + c) + str(8 - r)
            
        if debug_dir and best_square:
            os.makedirs(debug_dir, exist_ok=True)
            debug_img = img_bgr.copy()
            
            # Highlight the detected square
            self._draw_board_grid(debug_img, bx, by, sq_h, sq_w, default_color=(0, 255, 0), thickness=2, highlights={best_square: (255, 255, 255)})
            
            c = ord(best_square[0]) - ord('a')
            r = 8 - int(best_square[1])
            abs_x = int(bx + c * sq_w)
            abs_y = int(by + r * sq_h)
            
            cv2.putText(debug_img, f"Hover: {best_square}", (abs_x, abs_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imwrite(os.path.join(debug_dir, os.path.basename(image_path)), debug_img)
            
        print(f"  -> Found hover square: {best_square} with box score {max_box_score:.2f}")
        return best_square

    def extract_clocks(self, image_input, debug_dir=None):
        """Analyzes a static image or frame to extract the clock times and active player."""
        if isinstance(image_input, str):
            print(f"Extracting clocks in: {os.path.basename(image_input)}")
            img_bgr = cv2.imread(image_input)
            file_name = os.path.basename(image_input)
        else:
            img_bgr = image_input
            file_name = "clock_debug.png"
        
        bx, by, bh, bw, sq_h, sq_w = self.locate_board(img_bgr)
        
        # Clock UI is fixed relative to the board and slightly overhangs the board on the right.
        # These tighter ROIs hug the actual clock pill and exclude nearby player metadata.
        roi_x1 = max(0, int(bx + bw - sq_w * 1.18))
        roi_x2 = min(img_bgr.shape[1], int(bx + bw + sq_w * 0.05))

        top_roi_y1 = max(0, int(by - sq_h * 0.40))
        top_roi_y2 = max(0, int(by - sq_h * 0.08))

        bot_roi_y1 = min(img_bgr.shape[0], int(by + bh + sq_h * 0.08))
        bot_roi_y2 = min(img_bgr.shape[0], int(by + bh + sq_h * 0.40))
        
        top_bgr = img_bgr[top_roi_y1:top_roi_y2, roi_x1:roi_x2]
        bot_bgr = img_bgr[bot_roi_y1:bot_roi_y2, roi_x1:roi_x2]
        
        def get_white_bg_mask(roi_bgr):
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            return mask
            
        top_white_area = cv2.countNonZero(get_white_bg_mask(top_bgr))
        bot_white_area = cv2.countNonZero(get_white_bg_mask(bot_bgr))
        
        if top_white_area < 50 and bot_white_area < 50:
            active_player = None
        else:
            active_player = "white" if bot_white_area > top_white_area else "black"
        
        def parse_time_from_roi(roi_bgr, is_active):
            # Scale up to improve OCR accuracy on small digits
            roi_scaled = cv2.resize(roi_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(roi_scaled, cv2.COLOR_BGR2GRAY)
            
            if is_active:
                # Active: white bg, black text -> Threshold to pure black/white
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            else:
                # Inactive: dark bg, light text -> Invert to get black text on white bg
                _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
                
            # OCR configuration: Page Segmentation Mode 7 (single text line)
            # Adding a space to the whitelist prevents icons from fusing with the time digits
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789: '
            text = pytesseract.image_to_string(thresh, config=custom_config).strip()
            
            # Match strict time format first (ignores disconnected noise like '2 1:31:28')
            match = re.search(r'\d{1,2}:\d{2}(?::\d{2})?', text)
            if match:
                return match.group(0)
            return re.sub(r'[^0-9:]', '', text)

        black_time = parse_time_from_roi(top_bgr, active_player == "black")
        white_time = parse_time_from_roi(bot_bgr, active_player == "white")
        
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            debug_img = img_bgr.copy()
            top_color = (0, 255, 0) if active_player == "black" else (0, 0, 255)
            bot_color = (0, 255, 0) if active_player == "white" else (0, 0, 255)
            cv2.rectangle(debug_img, (roi_x1, top_roi_y1), (roi_x2, top_roi_y2), top_color, 3)
            cv2.rectangle(debug_img, (roi_x1, bot_roi_y1), (roi_x2, bot_roi_y2), bot_color, 3)
            cv2.imwrite(os.path.join(debug_dir, file_name), debug_img)
            
        # Only print if we are processing a file directly, else keep loop clean
        if isinstance(image_input, str):
            print(f"  -> Active: {active_player}, White: {white_time}, Black: {black_time}")
        return active_player, white_time, black_time

if __name__ == "__main__":
    video = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_games_short\7 plies\7 plies.mp4"
    board_asset = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\board\board.png"
    pgn = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_games_short\7 plies\game.pgn"
    
    extractor = ChessVideoExtractor(board_asset)
    extracted, expected = extractor.extract_moves_from_video(video, pgn)

    print("\n--- UNIT TEST RESULTS ---")
    assert extracted == expected, f"\nFAIL: Extracted moves do not match.\nExtracted: {extracted}\nExpected:  {expected}"
    print("PASS: Extracted moves perfectly match the expected 7 plies from the PGN.")
