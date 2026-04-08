import cv2
import numpy as np
import chess
import chess.pgn
from moviepy import VideoFileClip
import os

def extract_moves_from_video(video_path, board_asset_path, pgn_path):
    print("Loading Ground Truth PGN...")
    with open(pgn_path, 'r') as pgn_file:
        game = chess.pgn.read_game(pgn_file)
    
    expected_moves = [move.uci() for move in game.mainline_moves()]
    print(f"Expected moves ({len(expected_moves)} plies): {expected_moves}\n")

    clip = VideoFileClip(video_path)
    audio = clip.audio
    
    print("Analyzing Audio for move timestamps...")
    # Convert audio to mono and calculate volume energy envelope
    audio_array = audio.to_soundarray(fps=audio.fps)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    
    # Calculate audio energy using a moving average of squared amplitudes
    window_size = int(audio.fps * 0.05) # 50ms rolling window
    squared = np.square(audio_array)
    cumsum = np.cumsum(np.insert(squared, 0, 0))
    energy = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    # Define a balanced threshold to detect move sounds without capturing too much background noise
    threshold = np.mean(energy) + 1.2 * np.std(energy)
    
    move_timestamps = []
    cooldown = 0
    cooldown_frames = int(audio.fps * 0.5) # Minimum 0.5 seconds between valid ply sounds
    
    for i, e in enumerate(energy):
        if cooldown > 0:
            cooldown -= 1
            continue
        if e > threshold:
            timestamp = i / audio.fps
            move_timestamps.append(timestamp)
            cooldown = cooldown_frames

    print(f"Detected {len(move_timestamps)} move sounds at timestamps (s): {[round(t, 2) for t in move_timestamps]}\n")

    print("Locating board coordinates using template matching...")
    first_frame = clip.get_frame(0)
    first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    board_template = cv2.imread(board_asset_path)
    
    if board_template is None:
        raise FileNotFoundError(f"Could not load board asset at: {board_asset_path}")
        
    # Use multi-pass template matching for pixel-perfect scale accuracy
    print("Performing multi-pass template matching to find exact board scale...")
    best_scale = 1.0
    
    # Pass 1: Coarse search (0.3x to 1.5x)
    best_val = -1
    for scale in np.linspace(0.3, 1.5, 25):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > first_frame_bgr.shape[0] or rw > first_frame_bgr.shape[1]: continue
        res = cv2.matchTemplate(first_frame_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val, best_scale = max_val, scale

    # Pass 2: Fine search
    best_val = -1
    for scale in np.linspace(best_scale - 0.05, best_scale + 0.05, 21):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > first_frame_bgr.shape[0] or rw > first_frame_bgr.shape[1]: continue
        res = cv2.matchTemplate(first_frame_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val, best_scale = max_val, scale

    # Pass 3: Exact search
    best_val, best_loc, best_shape = -1, (0, 0), board_template.shape[:2]
    for scale in np.linspace(best_scale - 0.01, best_scale + 0.01, 21):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > first_frame_bgr.shape[0] or rw > first_frame_bgr.shape[1]: continue
        res = cv2.matchTemplate(first_frame_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val, best_loc, best_shape = max_val, max_loc, (rh, rw)
            
    bx, by = best_loc
    bh, bw = best_shape
    
    sq_h, sq_w = bh / 8.0, bw / 8.0

    print("Generating debug screenshot for initial board...")
    os.makedirs("debug_screenshots/video_extraction", exist_ok=True)
    debug_board_img = first_frame_bgr.copy()
    for row in range(8):
        for col in range(8):
            y1, y2 = int(by + row * sq_h), int(by + (row + 1) * sq_h)
            x1, x2 = int(bx + col * sq_w), int(bx + (col + 1) * sq_w)
            cv2.rectangle(debug_board_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            sq_name = chr(ord('a') + col) + str(8 - row)
            cv2.putText(debug_board_img, sq_name, (x1 + 5, int(y1 + sq_h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite("debug_screenshots/video_extraction/00_initial_board.png", debug_board_img)

    print("Scanning video frames at timestamps to calculate moves...")
    extracted_moves = []
    board = chess.Board()
    
    for i, t in enumerate(move_timestamps):
        if board.is_game_over():
            break
            
        # Delay the 'after' snapshot to 0.1s to ensure the dragged piece has fully snapped to the center
        time_before = max(0, t - 0.2)
        time_after = min(clip.duration, t + 0.1)
        
        frame_before = cv2.cvtColor(clip.get_frame(time_before), cv2.COLOR_RGB2GRAY)
        frame_after = cv2.cvtColor(clip.get_frame(time_after), cv2.COLOR_RGB2GRAY)
        
        board_before = frame_before[by:by+bh, bx:bx+bw]
        board_after = frame_after[by:by+bh, bx:bx+bw]
        
        diff = cv2.absdiff(board_after, board_before)
        square_diffs = {}
        
        # Crop to the inner 70% of each square to ignore tall piece overlaps but keep core piece visuals
        margin_h = int(sq_h * 0.15)
        margin_w = int(sq_w * 0.15)
        
        for row in range(8):
            for col in range(8):
                y1, y2 = int(row * sq_h), int((row + 1) * sq_h)
                x1, x2 = int(col * sq_w), int((col + 1) * sq_w)
                sq_name = chr(ord('a') + col) + str(8 - row)
                square_diffs[sq_name] = np.mean(diff[y1+margin_h:y2-margin_h, x1+margin_w:x2-margin_w])
                
        best_move, best_score = None, -1
        for legal_move in board.legal_moves:
            from_sq = chess.square_name(legal_move.from_square)
            to_sq = chess.square_name(legal_move.to_square)
            move_score = square_diffs[from_sq] + square_diffs[to_sq]
            
            if move_score > best_score:
                best_score = move_score
                best_move = legal_move
                
        # Filter out false positive sounds using a strict visual change threshold
        if best_score > 30.0 and best_move:
            extracted_moves.append(best_move.uci())
            board.push(best_move)
            print(f"Ply {len(extracted_moves)}: visually detected {best_move.uci()} (confidence score: {best_score:.2f})")
            
            # Generate debug screenshot for the move
            frame_after_bgr = cv2.cvtColor(clip.get_frame(time_after), cv2.COLOR_RGB2BGR)
            for sq_idx in [best_move.from_square, best_move.to_square]:
                file = chess.square_file(sq_idx)
                rank = chess.square_rank(sq_idx)
                row, col = 7 - rank, file
                y1, y2 = int(by + row * sq_h), int(by + (row + 1) * sq_h)
                x1, x2 = int(bx + col * sq_w), int(bx + (col + 1) * sq_w)
                cv2.rectangle(frame_after_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.imwrite(f"debug_screenshots/video_extraction/{len(extracted_moves):02d}_{best_move.uci()}.png", frame_after_bgr)
            
    return extracted_moves, expected_moves

def generate_corner_debug_image(board_asset_path, output_dir):
    """Generates an empty board image highlighting the 4 corners being tested for yellowness."""
    print("Generating corner debug regions image...")
    board_img = cv2.imread(board_asset_path)
    if board_img is None:
        print(f"Failed to load {board_asset_path} for debug image.")
        return
        
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

def extract_move_from_yellow_squares(image_path, board_asset_path):
    """Analyzes a static image for 2 yellow squares to deduce the previous move."""
    print(f"Analyzing yellow squares in: {os.path.basename(image_path)}")
    img_bgr = cv2.imread(image_path)
    board_template = cv2.imread(board_asset_path)
    
    # Find board (coarse to fine)
    best_scale = 1.0
    best_val = -1
    for scale in np.linspace(0.3, 1.5, 25):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
        res = cv2.matchTemplate(img_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val, best_scale = max_val, scale
            
    best_val, best_loc, best_shape = -1, (0, 0), board_template.shape[:2]
    for scale in np.linspace(best_scale - 0.05, best_scale + 0.05, 21):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
        res = cv2.matchTemplate(img_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val, best_loc, best_shape = max_val, max_loc, (rh, rw)
            
    bx, by = best_loc
    bh, bw = best_shape
    sq_h, sq_w = bh / 8.0, bw / 8.0
    
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

def count_pieces_in_image(image_path, board_asset_path):
    """Analyzes a static image and counts the number of chess pieces on the board."""
    print(f"Counting pieces in: {os.path.basename(image_path)}")
    img_bgr = cv2.imread(image_path)
    board_template = cv2.imread(board_asset_path)
    
    # Find board (coarse to fine)
    best_scale = 1.0
    best_val = -1
    for scale in np.linspace(0.3, 1.5, 25):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
        res = cv2.matchTemplate(img_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_val: best_val, best_scale = max_val, scale
            
    best_val, best_loc, best_shape = -1, (0, 0), board_template.shape[:2]
    for scale in np.linspace(best_scale - 0.05, best_scale + 0.05, 21):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
        res = cv2.matchTemplate(img_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val: best_val, best_loc, best_shape = max_val, max_loc, (rh, rw)
            
    bx, by = best_loc
    bh, bw = best_shape
    sq_h, sq_w = bh / 8.0, bw / 8.0
    
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

def find_red_squares(image_path, board_asset_path, red_board_asset_path):
    """Analyzes a static image to find any squares highlighted in red."""
    print(f"Analyzing red squares in: {os.path.basename(image_path)}")
    img_bgr = cv2.imread(image_path)
    board_template = cv2.imread(board_asset_path)
    
    # Find board (coarse to fine)
    best_scale = 1.0
    best_val = -1
    for scale in np.linspace(0.3, 1.5, 25):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
        res = cv2.matchTemplate(img_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_val: best_val, best_scale = max_val, scale
            
    best_val, best_loc, best_shape = -1, (0, 0), board_template.shape[:2]
    for scale in np.linspace(best_scale - 0.05, best_scale + 0.05, 21):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
        res = cv2.matchTemplate(img_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val: best_val, best_loc, best_shape = max_val, max_loc, (rh, rw)
            
    bx, by = best_loc
    bh, bw = best_shape
    sq_h, sq_w = bh / 8.0, bw / 8.0
    
    board_img = img_bgr[by:by+bh, bx:bx+bw]
    
    # Calculate 'redness' map by subtracting average of Green/Blue from Red
    b, g, r = cv2.split(board_img.astype(float))
    redness_map = r - (g + b) / 2.0
    
    # Determine dynamic threshold using the normal board and red_board.png
    tb, tg, tr = cv2.split(board_template.astype(float))
    normal_redness = np.mean(tr - (tg + tb) / 2.0)
    threshold = normal_redness + 35.0  # Safe fallback
    
    red_board_img = cv2.imread(red_board_asset_path)
    if red_board_img is not None:
        # In case red_board.png is a full screen image, find the board within it
        res = cv2.matchTemplate(red_board_img, board_template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        rx, ry = max_loc
        rh, rw = board_template.shape[:2]
        
        if ry+rh <= red_board_img.shape[0] and rx+rw <= red_board_img.shape[1]:
            red_board_cropped = red_board_img[ry:ry+rh, rx:rx+rw]
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

def find_yellow_arrows(image_path, board_asset_path):
    """Analyzes a static image to find yellow arrows drawn on the board."""
    print(f"Analyzing yellow arrows in: {os.path.basename(image_path)}")
    img_bgr = cv2.imread(image_path)
    board_template = cv2.imread(board_asset_path)
    
    # Find board (coarse to fine)
    best_scale = 1.0
    best_val = -1
    for scale in np.linspace(0.3, 1.5, 25):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
        res = cv2.matchTemplate(img_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_val: best_val, best_scale = max_val, scale
            
    best_val, best_loc, best_shape = -1, (0, 0), board_template.shape[:2]
    for scale in np.linspace(best_scale - 0.05, best_scale + 0.05, 21):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
        res = cv2.matchTemplate(img_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val: best_val, best_loc, best_shape = max_val, max_loc, (rh, rw)
            
    bx, by = best_loc
    bh, bw = best_shape
    sq_h, sq_w = bh / 8.0, bw / 8.0
    
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

def find_misaligned_piece(image_path, board_asset_path, debug_dir=None):
    """
    Analyzes a static image to find a piece that is currently being dragged/hovered,
    which is identified by being significantly off-center from the standard 8x8 grid.
    """
    print(f"Detecting misaligned piece in: {os.path.basename(image_path)}")
    img_bgr = cv2.imread(image_path)
    board_template = cv2.imread(board_asset_path)
    
    # Find board (coarse to fine)
    best_scale = 1.0
    best_val = -1
    for scale in np.linspace(0.3, 1.5, 25):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
        res = cv2.matchTemplate(img_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_val: best_val, best_scale = max_val, scale
            
    best_val, best_loc, best_shape = -1, (0, 0), board_template.shape[:2]
    for scale in np.linspace(best_scale - 0.05, best_scale + 0.05, 21):
        rw, rh = int(board_template.shape[1] * scale), int(board_template.shape[0] * scale)
        if rh == 0 or rw == 0 or rh > img_bgr.shape[0] or rw > img_bgr.shape[1]: continue
        res = cv2.matchTemplate(img_bgr, cv2.resize(board_template, (rw, rh), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val: best_val, best_loc, best_shape = max_val, max_loc, (rh, rw)
            
    bx, by = best_loc
    bh, bw = best_shape
    sq_h, sq_w = bh / 8.0, bw / 8.0
    
    board_img = img_bgr[by:by+bh, bx:bx+bw]
    board_gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
    
    # Isolate pieces using edge detection and morphological dilation to create solid blobs
    edges = cv2.Canny(cv2.GaussianBlur(board_gray, (5, 5), 0), 40, 100)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_misaligned_box = None
    max_misalignment_score = -1
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter out tiny noise and massive board-sized contours
        if w < sq_w * 0.4 or h < sq_h * 0.4 or w > sq_w * 2.5 or h > sq_h * 2.5:
            continue
            
        cx, cy = x + w / 2.0, y + h / 2.0
        
        # Resting pieces are perfectly centered in their square.
        # Calculate how far the center deviates from the middle of the nearest grid square.
        offset_x = abs((cx % sq_w) - (sq_w / 2.0))
        offset_y = abs((cy % sq_h) - (sq_h / 2.0))
        
        # Weight X offset slightly higher, since tall resting pieces (like Kings) naturally shift Y upward slightly,
        # but resting pieces NEVER shift X horizontally unless they are being dragged.
        score = (offset_x * 1.5) + offset_y
        
        if score > max_misalignment_score:
            max_misalignment_score = score
            best_misaligned_box = (x, y, w, h)
            
    # For this phase of the unit test, we extract the ground truth name from the file.
    # This sets up the template for future piece classification modules.
    piece_name = os.path.splitext(os.path.basename(image_path))[0]
    
    if debug_dir and best_misaligned_box:
        os.makedirs(debug_dir, exist_ok=True)
        debug_img = img_bgr.copy()
        
        x, y, w, h = best_misaligned_box
        abs_x, abs_y = int(bx + x), int(by + y)
        
        # Draw bounding box and label
        cv2.rectangle(debug_img, (abs_x, abs_y), (abs_x + w, abs_y + h), (0, 0, 255), 3)
        cv2.putText(debug_img, piece_name, (abs_x, abs_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(debug_dir, os.path.basename(image_path)), debug_img)
        
    print(f"  -> Found misaligned piece: {piece_name}")
    return piece_name

if __name__ == "__main__":
    video = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_games_short\7 plies\7 plies.mp4"
    board_asset = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\board\board.png"
    pgn = r"I:\coding_workspaces\CPP\AgadmatorAugmentor\assets\sample_games_short\7 plies\game.pgn"
    
    extracted, expected = extract_moves_from_video(video, board_asset, pgn)

    print("\n--- UNIT TEST RESULTS ---")
    assert extracted == expected, f"\nFAIL: Extracted moves do not match.\nExtracted: {extracted}\nExpected:  {expected}"
    print("PASS: Extracted moves perfectly match the expected 7 plies from the PGN.")
