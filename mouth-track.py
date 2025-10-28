import cv2
import mediapipe as mp
import numpy as np
import random
import time

# --- Configuration ---
REQUESTED_CAM_WIDTH = 1280
REQUESTED_CAM_HEIGHT = 720

# UI Dimensions
BANNER_TOP_HEIGHT = 70
BANNER_BOTTOM_HEIGHT = 60
UI_BORDER_THICKNESS = 10

# UI Colors
UI_BORDER_COLOR = (50, 50, 50)
BANNER_BG_COLOR = (70, 70, 70)
TEXT_COLOR_UI_BANNER = (220, 220, 220)
TEXT_COLOR_GAME = (255, 255, 255)
WIN_POPUP_BG_COLOR = (200, 255, 200)
WIN_POPUP_BORDER_COLOR = (0, 150, 0)
LOSE_POPUP_MSG_COLOR = (100, 100, 255) # Blueish for lose
RESTART_QUIT_TEXT_COLOR = (220, 220, 220)

# Game Parameters
HEART_SIZE = (35, 35)
PLAYER_LIVES = 3
POINTS_TO_WIN = 1
REWARD_CODE = "LAYSFREE20"
FALLING_SPEED_MIN_INITIAL = 3
FALLING_SPEED_MAX_INITIAL = 8
OBJECT_SPAWN_RATE_INITIAL = 45 # Frames between spawns
CHIP_SCORE = 1
CHIP_COLOR_FALLBACK = (0, 220, 220)
ROCK_COLOR_FALLBACK = (100, 100, 100)
ROCK_SPAWN_CHANCE_INITIAL = 0.35 # Probability of spawning a rock (0.35 = 35%)

# Mouth Tracking Parameters
MOUTH_AR_THRESHOLD = 0.30
MOUTH_CATCH_RADIUS_FACTOR = 0.35

# Asset Paths
PACKET_IMAGE_PATH = 'assets/win.png'
CHIP_IMAGE_PATH = 'assets/chip.png'
ROCK_IMAGE_PATH = 'assets/rock.png'
LIFE_IMAGE_PATH = 'assets/life.png'

# --- Global Asset Storage ---
packet_original_for_popup = None
chip_img = None
rock_img = None
life_icon_img = None

# --- Mediapipe Setup (initialized once) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Asset Loading Function ---
def load_and_resize_asset(path, target_size=None, asset_name="Asset"):
    """Loads an image asset, resizes it if target_size is provided, and handles errors."""
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: {asset_name} image not found at {path}. Using fallback if available.")
            return None
        print(f"{asset_name} image loaded from {path}.")
        if target_size:
            return cv2.resize(img, target_size)
        return img
    except Exception as e:
        print(f"Error loading/resizing {asset_name} from {path}: {e}")
        return None

# Load assets immediately
packet_img_orig_load = load_and_resize_asset(PACKET_IMAGE_PATH, asset_name="Chips Packet")
if packet_img_orig_load is not None:
    packet_original_for_popup = packet_img_orig_load.copy()

chip_img = load_and_resize_asset(CHIP_IMAGE_PATH, target_size=(40, 40), asset_name="Chip")
rock_img = load_and_resize_asset(ROCK_IMAGE_PATH, target_size=(45, 45), asset_name="Rock")
life_icon_img = load_and_resize_asset(LIFE_IMAGE_PATH, target_size=HEART_SIZE, asset_name="Life Icon")

# --- Helper Functions for Drawing ---
def draw_text_with_outline(img, text, pos, font, scale, text_color, outline_color, thickness, outline_thickness=1):
    """Draws text with a configurable outline for better visibility."""
    cv2.putText(img, text, pos, font, scale, outline_color, thickness + outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, text_color, thickness, cv2.LINE_AA)

def overlay_image_alpha(background_img, overlay_img_bgra, x, y):
    """Overlays an image with an alpha channel onto a background image."""
    if overlay_img_bgra is None:
        return

    h_overlay, w_overlay = overlay_img_bgra.shape[:2]
    h_background, w_background = background_img.shape[:2]

    # Convert coordinates to integers
    x, y = int(x), int(y)

    # Calculate region of interest in background
    x1_bg = max(x, 0)
    y1_bg = max(y, 0)
    x2_bg = min(x + w_overlay, w_background)
    y2_bg = min(y + h_overlay, h_background)

    # Calculate region of interest in overlay
    x1_ov = max(0, -x)
    y1_ov = max(0, -y)
    x2_ov = x1_ov + (x2_bg - x1_bg)
    y2_ov = y1_ov + (y2_bg - y1_bg)

    # Check for empty intersection
    if x2_bg <= x1_bg or y2_bg <= y1_bg:
        return

    roi_bg = background_img[y1_bg:y2_bg, x1_bg:x2_bg]
    roi_ov = overlay_img_bgra[y1_ov:y2_ov, x1_ov:x2_ov]

    if roi_ov.size == 0:
        return

    if roi_ov.shape[2] == 4:  # Overlay image has an alpha channel
        alpha = roi_ov[:, :, 3] / 255.0
        alpha_exp = np.dstack([alpha] * 3) # Expand alpha to 3 channels for broadcasting

        # Blend the images
        # Ensure dimensions match for element-wise multiplication
        if roi_bg.shape[:2] == alpha_exp.shape[:2]:
            roi_bg[:] = roi_bg * (1 - alpha_exp) + roi_ov[:, :, :3] * alpha_exp
        else:
            # Fallback for unexpected dimension mismatch - should ideally not happen
            print("Warning: ROI dimensions mismatch in overlay_image_alpha.")
            pass # Skip overlay for this region
    elif roi_ov.shape[2] == 3 and roi_bg.shape == roi_ov.shape:  # Overlay is RGB, no alpha
        roi_bg[:] = roi_ov

# --- Game Object Classes ---
class FallingObject:
    """Represents a falling object (chip or rock) in the game."""
    def __init__(self, x, y, speed, obj_type, image=None, radius=20, active=True):
        self.x = x
        self.y = y
        self.speed = speed
        self.obj_type = obj_type
        self.image = image
        self.radius = radius # Fallback radius if no image
        self.active = active # For object pooling

        if self.image is not None:
            self.height, self.width = self.image.shape[:2]
        else:
            self.width, self.height = radius * 2, radius * 2 # Default size if no image

        # Game-specific attributes
        if self.obj_type == "chip":
            self.score_value = CHIP_SCORE
            self.damage_value = 0
        elif self.obj_type == "rock":
            self.score_value = 0
            self.damage_value = 1
        else: # Default for unknown types
            self.score_value = 0
            self.damage_value = 0

    def fall(self):
        """Updates the object's vertical position based on its speed."""
        if self.active:
            self.y += self.speed

    def draw(self, frame):
        """Draws the object on the given frame."""
        if not self.active:
            return

        dx = int(self.x - self.width // 2)
        dy = int(self.y - self.height // 2)

        if self.image is not None:
            overlay_image_alpha(frame, self.image, dx, dy)
        else:
            # Fallback to drawing a circle if image is not loaded
            color = CHIP_COLOR_FALLBACK if self.obj_type == "chip" else ROCK_COLOR_FALLBACK
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, color, -1)

    def is_off_screen(self, h):
        """Checks if the object has fallen off the bottom of the screen."""
        return (self.y - self.height // 2) > h

    def get_rect(self):
        """Returns the bounding box of the object (x, y, width, height)."""
        return (self.x - self.width // 2, self.y - self.height // 2, self.width, self.height)

# --- Camera Initialization ---
def find_working_camera(req_width, req_height, max_to_check=3):
    """Finds and initializes a working camera with the requested resolution."""
    print("Attempting to find a suitable camera...")
    for i in range(max_to_check):
        cap_test = cv2.VideoCapture(i)
        print(f"Trying camera {i}...")
        if cap_test.isOpened():
            cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, req_width)
            cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, req_height)
            actual_w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))

            ret, frame = cap_test.read()
            if ret and frame is not None:
                # Simple check for non-black frames
                if np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 10:
                    print(f"Success: Camera {i} found at {actual_w}x{actual_h}.")
                    return cap_test, actual_w, actual_h
                else:
                    print(f"Info: Camera {i} ({actual_w}x{actual_h}) produced a black screen.")
            else:
                print(f"Info: Camera {i} ({actual_w}x{actual_h}) failed to read a frame.")
            cap_test.release()
        else:
            print(f"Info: Camera {i} could not be opened.")
    print("Error: No working camera found. Please ensure a camera is connected and not in use by another application.")
    return None, 0, 0

# --- Main Game Loop ---
def run_game():
    """Main function to run the Catch and Win game."""
    cap, actual_cam_width, actual_cam_height = find_working_camera(REQUESTED_CAM_WIDTH, REQUESTED_CAM_HEIGHT)
    if cap is None:
        err_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err_img, "No working camera found!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(err_img, "Press any key to exit.", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow("Camera Error", err_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Calculate game area and UI frame dimensions
    GAME_AREA_X_OFFSET = UI_BORDER_THICKNESS
    GAME_AREA_Y_OFFSET = BANNER_TOP_HEIGHT + UI_BORDER_THICKNESS
    UI_FRAME_WIDTH = actual_cam_width + 2 * UI_BORDER_THICKNESS
    UI_FRAME_HEIGHT = actual_cam_height + BANNER_TOP_HEIGHT + BANNER_BOTTOM_HEIGHT + 2 * UI_BORDER_THICKNESS

    WINDOW_NAME = 'Catch and Win - Mouth Eater!'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Game State Variables
    score = 0
    lives = PLAYER_LIVES
    game_over = False
    game_won = False
    spawn_counter = 0

    # Object Pooling for falling objects
    falling_objects_pool = []
    MAX_POOL_SIZE = 50 # Max number of objects to keep in pool
    active_objects = []

    mouth_is_open_flag = False
    mouth_center_x, mouth_center_y = 0, 0
    current_mouth_catch_radius = 30

    # Difficulty settings (can be dynamically updated)
    current_fall_speed_min = FALLING_SPEED_MIN_INITIAL
    current_fall_speed_max = FALLING_SPEED_MAX_INITIAL
    current_spawn_rate = OBJECT_SPAWN_RATE_INITIAL
    rock_spawn_chance = ROCK_SPAWN_CHANCE_INITIAL

    # Pre-calculate text sizes for static UI elements
    title_txt = "Catch and Win!"
    title_text_size = cv2.getTextSize(title_txt, cv2.FONT_HERSHEY_TRIPLEX, 1.5, 3)[0]
    title_pos_x = (UI_FRAME_WIDTH - title_text_size[0]) // 2
    title_pos_y = (BANNER_TOP_HEIGHT + title_text_size[1]) // 2 + UI_BORDER_THICKNESS // 2

    instr_txt = f"Open your mouth to eat {POINTS_TO_WIN} chips!"
    instr_text_size = cv2.getTextSize(instr_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    instr_pos_x = (UI_FRAME_WIDTH - instr_text_size[0]) // 2
    instr_pos_y = UI_FRAME_HEIGHT - (BANNER_BOTTOM_HEIGHT + instr_text_size[1]) // 2 - UI_BORDER_THICKNESS + 10

    # Win/Lose Popup dimensions
    POPUP_WIDTH = 500
    POPUP_HEIGHT_WIN = 450
    POPUP_HEIGHT_LOSE = 150 # Smaller for lose screen

    win_popup_packet_resized = None
    if packet_original_for_popup is not None:
        pop_pack_w = 250
        pop_pack_h = int(pop_pack_w * (packet_original_for_popup.shape[0] / packet_original_for_popup.shape[1]))
        win_popup_packet_resized = cv2.resize(packet_original_for_popup, (pop_pack_w, pop_pack_h))

    while True:
        if not cap.isOpened():
            print("Error: Camera disconnected or failed.")
            break

        success, game_area_frame_raw = cap.read()
        if not success or game_area_frame_raw is None:
            print("Error: Failed to read frame from camera.")
            break

        game_area_frame = cv2.flip(game_area_frame_raw, 1) # Flip horizontally for selfie view

        # Process face mesh
        game_area_frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(game_area_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        game_area_frame.flags.writeable = True

        mouth_is_open_flag = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Landmarks for upper and lower lip (approximate for AR)
                y_upper_lip_px = face_landmarks.landmark[13].y * actual_cam_height
                y_lower_lip_px = face_landmarks.landmark[14].y * actual_cam_height
                # Landmarks for left and right corners of the mouth
                x_left_mouth_px = face_landmarks.landmark[61].x * actual_cam_width
                x_right_mouth_px = face_landmarks.landmark[291].x * actual_cam_width

                mouth_vertical_dist = abs(y_lower_lip_px - y_upper_lip_px)
                mouth_horizontal_dist = abs(x_right_mouth_px - x_left_mouth_px)

                if mouth_horizontal_dist > 0: # Avoid division by zero
                    mouth_aspect_ratio = mouth_vertical_dist / mouth_horizontal_dist
                    if mouth_aspect_ratio > MOUTH_AR_THRESHOLD:
                        mouth_is_open_flag = True

                mouth_center_x = (x_left_mouth_px + x_right_mouth_px) / 2
                mouth_center_y = (y_upper_lip_px + y_lower_lip_px) / 2
                current_mouth_catch_radius = (mouth_horizontal_dist * MOUTH_CATCH_RADIUS_FACTOR)

        # Game Logic
        if not game_over:
            spawn_counter += 1
            if spawn_counter >= current_spawn_rate:
                spawn_counter = 0

                obj_type = "chip" if random.random() > rock_spawn_chance else "rock"
                img_use = chip_img if obj_type == "chip" else rock_img
                
                # Try to get an object from the pool, otherwise create a new one
                new_obj = None
                for obj in falling_objects_pool:
                    if not obj.active:
                        new_obj = obj
                        break
                
                if new_obj:
                    # Reset existing object from pool
                    new_obj.x = random.randint(new_obj.width // 2, actual_cam_width - new_obj.width // 2) if actual_cam_width > new_obj.width else actual_cam_width // 2
                    new_obj.y = 0
                    new_obj.speed = random.randint(current_fall_speed_min, current_fall_speed_max)
                    new_obj.obj_type = obj_type
                    new_obj.image = img_use
                    new_obj.active = True
                    # Update dimensions if image changed (though usually constant for chip/rock)
                    if new_obj.image is not None:
                        new_obj.height, new_obj.width = new_obj.image.shape[:2]
                    else:
                        new_obj.width, new_obj.height = new_obj.radius * 2, new_obj.radius * 2
                    
                    new_obj.score_value = CHIP_SCORE if obj_type == "chip" else 0
                    new_obj.damage_value = 0 if obj_type == "chip" else 1

                else:
                    # Create a new object if pool is empty or full of active objects
                    cur_w = 40 # Default if no image
                    rad_new = 20
                    if img_use is not None:
                        cur_w = img_use.shape[1]
                    else:
                        cur_w = rad_new * 2

                    obj_x = actual_cam_width // 2
                    if actual_cam_width > cur_w:
                        obj_x = random.randint(cur_w // 2, actual_cam_width - cur_w // 2)
                    
                    new_obj = FallingObject(obj_x, 0,
                                            random.randint(current_fall_speed_min, current_fall_speed_max),
                                            obj_type, image=img_use, radius=rad_new, active=True)
                    
                    if len(falling_objects_pool) < MAX_POOL_SIZE:
                        falling_objects_pool.append(new_obj)
                
                if new_obj and new_obj.active: # Add to active list only if it's new or was reused
                    active_objects.append(new_obj)


            # Update and check collisions for active objects
            new_active_objects = []
            for obj in active_objects:
                if not obj.active: # Skip if already processed or deactivated
                    continue

                obj.fall()
                obj_rect_x, obj_rect_y, obj_rect_w, obj_rect_h = obj.get_rect()
                obj_center_x_fall = obj_rect_x + obj_rect_w / 2
                obj_center_y_fall = obj_rect_y + obj_rect_h / 2

                collided_with_mouth = False
                if mouth_is_open_flag and results.multi_face_landmarks:
                    # Simple circular collision detection
                    dist_sq = (obj_center_x_fall - mouth_center_x)**2 + \
                              (obj_center_y_fall - mouth_center_y)**2
                    if dist_sq < (current_mouth_catch_radius + max(obj_rect_w, obj_rect_h) / 2)**2: # Use max dimension for collision
                        collided_with_mouth = True

                if collided_with_mouth:
                    if obj.obj_type == "chip":
                        score += obj.score_value
                        if score >= POINTS_TO_WIN:
                            game_won = True
                            game_over = True
                    else: # rock
                        lives -= obj.damage_value
                        if lives <= 0:
                            lives = 0
                            game_over = True
                    obj.active = False # Deactivate for pooling
                elif obj.is_off_screen(actual_cam_height):
                    obj.active = False # Deactivate for pooling
                else:
                    new_active_objects.append(obj)
            
            active_objects = new_active_objects # Update active objects list

            # --- Optional Difficulty Scaling (Uncomment to activate) ---
            if score > 0 and score % 5 == 0: # Every 5 points
                current_fall_speed_min = min(15, FALLING_SPEED_MIN_INITIAL + score // 5)
                current_fall_speed_max = min(20, FALLING_SPEED_MAX_INITIAL + score // 5)
                current_spawn_rate = max(10, OBJECT_SPAWN_RATE_INITIAL - score // 5)
                rock_spawn_chance = min(0.75, ROCK_SPAWN_CHANCE_INITIAL + (score // 10) * 0.05)


        # Drawing all active objects
        for obj in active_objects:
            obj.draw(game_area_frame)
        
        # --- Constructing the UI Canvas ---
        ui_canvas = np.full((UI_FRAME_HEIGHT, UI_FRAME_WIDTH, 3), BANNER_BG_COLOR, dtype=np.uint8)

        # Draw main UI border
        cv2.rectangle(ui_canvas, (0, 0), (UI_FRAME_WIDTH - 1, UI_FRAME_HEIGHT - 1),
                      UI_BORDER_COLOR, UI_BORDER_THICKNESS * 2)

        # Place the game area frame into the UI canvas
        ui_canvas[GAME_AREA_Y_OFFSET:GAME_AREA_Y_OFFSET + actual_cam_height,
                  GAME_AREA_X_OFFSET:GAME_AREA_X_OFFSET + actual_cam_width] = game_area_frame

        # Draw UI elements
        # Title
        draw_text_with_outline(ui_canvas, title_txt, (title_pos_x, title_pos_y),
                               cv2.FONT_HERSHEY_TRIPLEX, 1.5, TEXT_COLOR_UI_BANNER, (20, 20, 20), 3, 2)

        # Instructions
        draw_text_with_outline(ui_canvas, instr_txt, (instr_pos_x, instr_pos_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR_UI_BANNER, (20, 20, 20), 2, 1)

        # Score display
        score_txt = f"Score: {score}"
        score_pos_x = GAME_AREA_X_OFFSET + 20
        score_pos_y = GAME_AREA_Y_OFFSET + 40 # Adjust for better visibility
        draw_text_with_outline(ui_canvas, score_txt, (score_pos_x, score_pos_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR_GAME, (0, 0, 0), 2)

        # Lives display
        heart_offset_x = GAME_AREA_X_OFFSET + actual_cam_width - (PLAYER_LIVES * (HEART_SIZE[0] + 10))
        heart_offset_y = GAME_AREA_Y_OFFSET + 15
        for i in range(lives):
            hx = heart_offset_x + i * (HEART_SIZE[0] + 5)
            if life_icon_img is not None:
                overlay_image_alpha(ui_canvas, life_icon_img, hx, heart_offset_y)
            else:
                # Fallback to drawing red circles for hearts
                cv2.circle(ui_canvas, (hx + HEART_SIZE[0] // 2, heart_offset_y + HEART_SIZE[1] // 2), HEART_SIZE[0] // 2, (0, 0, 220), -1)
                cv2.circle(ui_canvas, (hx + HEART_SIZE[0] // 2, heart_offset_y + HEART_SIZE[1] // 2), HEART_SIZE[0] // 2, (0, 0, 150), 2)


        # --- Game Over / Game Won Pop-up ---
        if game_over:
            # Darken the game area
            sub_img = ui_canvas[GAME_AREA_Y_OFFSET:GAME_AREA_Y_OFFSET + actual_cam_height,
                                GAME_AREA_X_OFFSET:GAME_AREA_X_OFFSET + actual_cam_width]
            res = cv2.addWeighted(sub_img, 0.2, np.full(sub_img.shape, (0, 0, 0), dtype=np.uint8), 0.8, 1.0)
            ui_canvas[GAME_AREA_Y_OFFSET:GAME_AREA_Y_OFFSET + actual_cam_height,
                      GAME_AREA_X_OFFSET:GAME_AREA_X_OFFSET + actual_cam_width] = res

            if game_won:
                popup_h = POPUP_HEIGHT_WIN
                popup_x = (UI_FRAME_WIDTH - POPUP_WIDTH) // 2
                popup_y = (UI_FRAME_HEIGHT - popup_h) // 2

                cv2.rectangle(ui_canvas, (popup_x, popup_y), (popup_x + POPUP_WIDTH, popup_y + popup_h),
                              WIN_POPUP_BG_COLOR, -1) # Background fill
                cv2.rectangle(ui_canvas, (popup_x, popup_y), (popup_x + POPUP_WIDTH, popup_y + popup_h),
                              WIN_POPUP_BORDER_COLOR, 3) # Border

                # "YOU WON!" text
                win_txt = "YOU WON!"
                wtsz = cv2.getTextSize(win_txt, cv2.FONT_HERSHEY_TRIPLEX, 2, 3)[0]
                draw_text_with_outline(ui_canvas, win_txt, (popup_x + (POPUP_WIDTH - wtsz[0]) // 2, popup_y + 60),
                                       cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 128, 0), (255, 255, 255), 3, 2)
                
                # Display final score
                final_score_txt = f"Final Score: {score}"
                fs_w, fs_h = cv2.getTextSize(final_score_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                draw_text_with_outline(ui_canvas, final_score_txt, (popup_x + (POPUP_WIDTH - fs_w) // 2, popup_y + 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), (255, 255, 255), 2)


                pack_image_y_start = popup_y + 120 # Adjust position based on score text
                if win_popup_packet_resized is not None:
                    pk_x = popup_x + (POPUP_WIDTH - win_popup_packet_resized.shape[1]) // 2
                    overlay_image_alpha(ui_canvas, win_popup_packet_resized, pk_x, pack_image_y_start)
                    reward_text_y_start = pack_image_y_start + win_popup_packet_resized.shape[0] + 20
                else:
                    cv2.putText(ui_canvas, "[CHIPS PACKET IMAGE]", (popup_x + 150, pack_image_y_start + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    reward_text_y_start = pack_image_y_start + 80

                # Reward text
                code_prompt_txt = "Use this code to redeem your prize:"
                cpt1sz = cv2.getTextSize(code_prompt_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                draw_text_with_outline(ui_canvas, code_prompt_txt,
                                       (popup_x + (POPUP_WIDTH - cpt1sz[0]) // 2, reward_text_y_start),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), (255, 255, 255), 2)

                # Reward code
                code_txt = REWARD_CODE
                cpt2sz = cv2.getTextSize(code_txt, cv2.FONT_HERSHEY_TRIPLEX, 1.2, 2)[0]
                draw_text_with_outline(ui_canvas, code_txt,
                                       (popup_x + (POPUP_WIDTH - cpt2sz[0]) // 2, reward_text_y_start + cpt1sz[1] + 20),
                                       cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 100, 0), (255, 255, 255), 2, 2)
            else: # Game Lost
                popup_h = POPUP_HEIGHT_LOSE
                popup_x = (UI_FRAME_WIDTH - POPUP_WIDTH) // 2
                popup_y = (UI_FRAME_HEIGHT - popup_h) // 2

                cv2.rectangle(ui_canvas, (popup_x, popup_y), (popup_x + POPUP_WIDTH, popup_y + popup_h),
                              (255, 200, 200), -1) # Reddish background
                cv2.rectangle(ui_canvas, (popup_x, popup_y), (popup_x + POPUP_WIDTH, popup_y + popup_h),
                              (0, 0, 150), 3) # Darker red border

                # "GAME OVER" text
                msg_txt = "GAME OVER"
                msg_w, msg_h = cv2.getTextSize(msg_txt, cv2.FONT_HERSHEY_TRIPLEX, 2.5, 3)[0]
                draw_text_with_outline(ui_canvas, msg_txt,
                                       (popup_x + (POPUP_WIDTH - msg_w) // 2, popup_y + msg_h + 30),
                                       cv2.FONT_HERSHEY_TRIPLEX, 2.5, LOSE_POPUP_MSG_COLOR, (0, 0, 0), 3, 2)
                
                # Display final score
                final_score_txt = f"Final Score: {score}"
                fs_w, fs_h = cv2.getTextSize(final_score_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                draw_text_with_outline(ui_canvas, final_score_txt, (popup_x + (POPUP_WIDTH - fs_w) // 2, popup_y + msg_h + 30 + fs_h + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), (255, 255, 255), 2)


            # Restart/Quit message (common to both win/lose)
            pr_msg = "Press 'R' to Restart or 'Q' to Quit"
            pr_msg_w, pr_msg_h = cv2.getTextSize(pr_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            
            # Position restart/quit message centrally below the popup
            pr_y = popup_y + popup_h + 40
            if pr_y + pr_msg_h > UI_FRAME_HEIGHT - UI_BORDER_THICKNESS - 10: # Avoid going off screen
                pr_y = UI_FRAME_HEIGHT - UI_BORDER_THICKNESS - 10 - pr_msg_h
            
            draw_text_with_outline(ui_canvas, pr_msg,
                                   ((UI_FRAME_WIDTH - pr_msg_w) // 2, pr_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, RESTART_QUIT_TEXT_COLOR, (0, 0, 0), 2)

            cv2.imshow(WINDOW_NAME, ui_canvas)
            key = cv2.waitKey(0) & 0xFF # Wait indefinitely for input

            if key == ord('r') or key == ord('R'):
                score = 0
                lives = PLAYER_LIVES
                game_over = False
                game_won = False
                active_objects = [] # Clear active objects
                # Reset all pooled objects to inactive for a fresh start
                for obj in falling_objects_pool:
                    obj.active = False
                spawn_counter = 0
                
                # Reset difficulty
                current_fall_speed_min = FALLING_SPEED_MIN_INITIAL
                current_fall_speed_max = FALLING_SPEED_MAX_INITIAL
                current_spawn_rate = OBJECT_SPAWN_RATE_INITIAL
                rock_spawn_chance = ROCK_SPAWN_CHANCE_INITIAL

                continue
            elif key == ord('q') or key == ord('Q'):
                break
        else:
            # Game is active, just display the frame
            cv2.imshow(WINDOW_NAME, ui_canvas)
            key = cv2.waitKey(1) & 0xFF # Wait 1ms for input

            if key == ord('q') or key == ord('Q'):
                break

    # Clean up
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if face_mesh: # Ensure face_mesh is closed if it was initialized
        face_mesh.close()

if __name__ == '__main__':
    run_game()