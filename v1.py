import cv2
import mediapipe as mp
import numpy as np
import random
import time
import pygame  # Added for sound
from PIL import Image, ImageFont, ImageDraw  # Added for custom fonts
import sys
import os

# --- Helper Function for PyInstaller ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Not running in a bundle
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# -------------------------------------

# --- Configuration ---
REQUESTED_CAM_WIDTH = 1280
REQUESTED_CAM_HEIGHT = 720

# UI Dimensions
BANNER_TOP_HEIGHT = 70
BANNER_BOTTOM_HEIGHT = 60
UI_BORDER_THICKNESS = 10

# UI Colors (Refined)
UI_BORDER_COLOR = (50, 50, 50)
BANNER_BG_COLOR = (70, 70, 70)
TEXT_COLOR_UI_BANNER = (255, 255, 255) # Brighter
TEXT_COLOR_GAME = (255, 255, 255)
TEXT_COLOR_OUTLINE = (0, 0, 0)
POPUP_BG_COLOR = (255, 255, 255) # Clean white
POPUP_BG_COLOR_LOSE = (40, 40, 40) # Dark
POPUP_BORDER_WIN = (0, 180, 0)
POPUP_BORDER_LOSE = (180, 0, 0)
POPUP_ALPHA = 0.85 # Transparency
HIT_FLASH_COLOR = (0, 0, 255) # Red flash
RESTART_QUIT_TEXT_COLOR = (230, 230, 230)

# Game Parameters
HEART_SIZE = (35, 35)
PLAYER_LIVES = 3
POINTS_TO_WIN = 20
REWARD_CODE = "LAYSFREE20"
FALLING_SPEED_MIN_INITIAL = 3
FALLING_SPEED_MAX_INITIAL = 8
OBJECT_SPAWN_RATE_INITIAL = 40
CHIP_SCORE = 1
CHIP_COLOR_FALLBACK = (0, 220, 220)
ROCK_COLOR_FALLBACK = (100, 100, 100)

# New Game Mechanics Parameters
ROCK_SPAWN_CHANCE_INITIAL = 0.35
POWERUP_HEART_SPAWN_CHANCE = 0.05 # 5% chance
POWERUP_SLOWMO_SPAWN_CHANCE = 0.03 # 3% chance
SLOWMO_DURATION_FRAMES = 300 # 5 seconds at 60fps
SLOWMO_FACTOR = 0.5
COMBO_BONUS_THRESHOLD = 5 # Bonus point every 5 combos
FLOATING_TEXT_LIFESPAN = 45 # Frames
POPUP_ANIMATION_FRAMES = 30 # ~0.5 seconds for ease-in

# Mouth Tracking Parameters
MOUTH_AR_THRESHOLD = 0.30
MOUTH_CATCH_RADIUS_FACTOR = 0.35

# Asset Paths
PACKET_IMAGE_PATH = resource_path('assets/win.png')
CHIP_IMAGE_PATH = resource_path('assets/chip.png')
ROCK_IMAGE_PATH = resource_path('assets/rock.png')
LIFE_IMAGE_PATH = resource_path('assets/life.png')
SLOWMO_IMAGE_PATH = resource_path('assets/slowmo.png')
GAME_FONT_PATH = resource_path('assets/font/game_font.ttf')

# Sound Paths
CRUNCH_SOUND_PATH = resource_path('assets/sounds/crunch.wav')
ROCK_SOUND_PATH = resource_path('assets/sounds/rock_hit.wav')
POWERUP_SOUND_PATH = resource_path('assets/sounds/powerup.wav')
SLOWMO_SOUND_PATH = resource_path('assets/sounds/slowmo.wav')
WIN_SOUND_PATH = resource_path('assets/sounds/win.wav')
LOSE_SOUND_PATH = resource_path('assets/sounds/lose.wav')

# --- Global Asset Storage ---
packet_original_for_popup = None
chip_img = None
rock_img = None
life_icon_img = None
slowmo_img = None
title_font = None # PIL Font for main titles
ui_font = None    # PIL Font for general UI
ui_font_small = None # PIL Font for smaller UI text
sounds = {}

# --- Mediapipe Setup (initialized once) ---
# We now need FaceMesh (for playing) and Hands (for starting)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# --- Asset Loading Functions ---

class DummySound:
    """A dummy class that does nothing, to prevent crashes if sounds fail to load."""
    def play(self):
        pass

def load_sound(path, sound_name):
    """Loads a single sound effect and returns a DummySound on failure."""
    try:
        sound = pygame.mixer.Sound(path)
        print(f"Sound '{sound_name}' loaded from {path}.")
        return sound
    except Exception as e:
        print(f"Warning: Sound '{sound_name}' not found at {path}. {e}")
        return DummySound()

def load_game_assets():
    """Loads all images, sounds, and fonts."""
    global packet_original_for_popup, chip_img, rock_img, life_icon_img, slowmo_img
    global title_font, ui_font, ui_font_small, sounds

    # --- Load Images ---
    packet_img_orig_load = load_and_resize_asset(PACKET_IMAGE_PATH, asset_name="Chips Packet")
    if packet_img_orig_load is not None:
        packet_original_for_popup = packet_img_orig_load.copy()

    chip_img = load_and_resize_asset(CHIP_IMAGE_PATH, target_size=(40, 40), asset_name="Chip")
    rock_img = load_and_resize_asset(ROCK_IMAGE_PATH, target_size=(45, 45), asset_name="Rock")
    life_icon_img = load_and_resize_asset(LIFE_IMAGE_PATH, target_size=HEART_SIZE, asset_name="Life Icon")
    slowmo_img = load_and_resize_asset(SLOWMO_IMAGE_PATH, target_size=(40, 40), asset_name="Slow-Mo")
    
    # --- Load Fonts ---
    try:
        title_font = ImageFont.truetype(GAME_FONT_PATH, 48) # For "Catch and Win!", "YOU WON!"
        ui_font = ImageFont.truetype(GAME_FONT_PATH, 24)   # For instructions, reward code, game over
        ui_font_small = ImageFont.truetype(GAME_FONT_PATH, 18) # For "Use this code..."
        print(f"Fonts loaded from {GAME_FONT_PATH}.")
    except IOError:
        print(f"Warning: Custom font not found at {GAME_FONT_PATH}. Using default.")
        title_font = ImageFont.load_default()
        ui_font = ImageFont.load_default()
        ui_font_small = ImageFont.load_default()

    # --- Load Sounds ---
    pygame.mixer.init()
    sounds['crunch'] = load_sound(CRUNCH_SOUND_PATH, 'crunch')
    sounds['rock_hit'] = load_sound(ROCK_SOUND_PATH, 'rock_hit')
    sounds['powerup'] = load_sound(POWERUP_SOUND_PATH, 'powerup')
    sounds['slowmo'] = load_sound(SLOWMO_SOUND_PATH, 'slowmo')
    sounds['win'] = load_sound(WIN_SOUND_PATH, 'win')
    sounds['lose'] = load_sound(LOSE_SOUND_PATH, 'lose')

def load_and_resize_asset(path, target_size=None, asset_name="Asset"):
    """Loads an image asset, resizes it if target_size is provided, and handles errors."""
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: {asset_name} image not found at {path}. Using fallback if available.")
            return None
        if target_size:
            return cv2.resize(img, target_size)
        return img
    except Exception as e:
        print(f"Error loading/resizing {asset_name} from {path}: {e}")
        return None

# --- Helper Functions for Drawing ---

def draw_text_with_outline(img, text, pos, font, scale, text_color, outline_color, thickness, outline_thickness=1):
    """Draws text with a configurable outline for better visibility using cv2.putText."""
    cv2.putText(img, text, pos, font, scale, outline_color, thickness + outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, text_color, thickness, cv2.LINE_AA)

def draw_pil_text(cv2_img, text, pos, font_pil, fill_color):
    """Converts to PIL, draws high-quality text, converts back."""
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=font_pil, fill=fill_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def get_pil_text_size(text, font_pil):
    """Calculates text size using PIL for accurate centering."""
    if text == "": return (0, 0)
    bbox = font_pil.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def draw_transparent_rectangle(img, rect_coords, color, alpha):
    """Draws a semi-transparent rectangle."""
    x, y, w, h = rect_coords
    if w <= 0 or h <= 0: return # Don't draw if invisible
    try:
        sub_img = img[y:y+h, x:x+w]
        overlay = np.full(sub_img.shape, color, dtype=np.uint8)
        res = cv2.addWeighted(overlay, alpha, sub_img, 1 - alpha, 0)
        img[y:y+h, x:x+w] = res
    except Exception as e:
        # Avoid crash if rect is off-screen or dimensions are mismatched
        pass 

def overlay_image_alpha(background_img, overlay_img_bgra, x, y):
    """Overlays an image with an alpha channel onto a background image."""
    if overlay_img_bgra is None:
        return

    h_overlay, w_overlay = overlay_img_bgra.shape[:2]
    if w_overlay <= 0 or h_overlay <= 0: return # Don't draw if invisible

    h_background, w_background = background_img.shape[:2]
    x, y = int(x), int(y)

    x1_bg = max(x, 0)
    y1_bg = max(y, 0)
    x2_bg = min(x + w_overlay, w_background)
    y2_bg = min(y + h_overlay, h_background)

    x1_ov = max(0, -x)
    y1_ov = max(0, -y)
    x2_ov = x1_ov + (x2_bg - x1_bg)
    y2_ov = y1_ov + (y2_bg - y1_bg)

    if x2_bg <= x1_bg or y2_bg <= y1_bg:
        return

    roi_bg = background_img[y1_bg:y2_bg, x1_bg:x2_bg]
    roi_ov = overlay_img_bgra[y1_ov:y2_ov, x1_ov:x2_ov]

    if roi_ov.size == 0 or roi_bg.size == 0:
        return

    if roi_ov.shape[2] == 4:
        alpha = roi_ov[:, :, 3] / 255.0
        alpha_exp = np.dstack([alpha] * 3)

        if roi_bg.shape[:2] == alpha_exp.shape[:2]:
            roi_bg[:] = roi_bg * (1 - alpha_exp) + roi_ov[:, :, :3] * alpha_exp
        else:
            # Fallback for dimension mismatch (crop if necessary)
            min_h = min(roi_bg.shape[0], roi_ov.shape[0])
            min_w = min(roi_bg.shape[1], roi_ov.shape[1])
            
            roi_bg_cropped = roi_bg[:min_h, :min_w, :]
            roi_ov_cropped = roi_ov[:min_h, :min_w, :3]
            alpha_exp_cropped = alpha_exp[:min_h, :min_w, :]

            roi_bg_cropped[:] = roi_bg_cropped * (1 - alpha_exp_cropped) + roi_ov_cropped * alpha_exp_cropped
            img[y1_bg:y1_bg+min_h, x1_bg:x1_bg+min_w] = roi_bg_cropped
            
    elif roi_ov.shape[2] == 3 and roi_bg.shape == roi_ov.shape: # No alpha channel, simple overlay
        roi_bg[:] = roi_ov

# --- Gesture Helper ---
def is_thumbs_up(hand_landmarks):
    """Checks if a hand is in a 'thumbs up' gesture. More robust."""
    try:
        lm = hand_landmarks.landmark
        
        # Landmark indices
        thumb_tip = 4
        thumb_ip = 3
        thumb_mcp = 2
        
        index_tip = 8
        index_mcp = 5
        middle_tip = 12
        middle_mcp = 9
        ring_tip = 16
        ring_mcp = 13
        pinky_tip = 20
        pinky_mcp = 17

        # Thumb is up: Tip Y is less than (higher on screen) than IP and MCP
        is_thumb_up = (lm[thumb_tip].y < lm[thumb_ip].y) and (lm[thumb_tip].y < lm[thumb_mcp].y)
        
        # Fingers are down: Tips Y are greater than (lower on screen) than MCPs
        is_index_down = lm[index_tip].y > lm[index_mcp].y
        is_middle_down = lm[middle_tip].y > lm[middle_mcp].y
        is_ring_down = lm[ring_tip].y > lm[ring_mcp].y
        is_pinky_down = lm[pinky_tip].y > lm[pinky_mcp].y
        
        are_fingers_down = is_index_down and is_middle_down and is_ring_down and is_pinky_down
                            
        return is_thumb_up and are_fingers_down
    except Exception as e:
        return False

# --- Game Object Classes ---
class FallingObject:
    """Represents a falling object (chip, rock, or power-up)."""
    def __init__(self, x, y, speed, obj_type, image=None, radius=20, active=True):
        self.x = x
        self.y = y
        self.speed = speed
        self.obj_type = obj_type
        self.image = image
        self.radius = radius
        self.active = active

        if self.image is not None:
            self.height, self.width = self.image.shape[:2]
        else:
            self.width, self.height = radius * 2, radius * 2

        self.set_type(obj_type) # Set score/damage values

    def set_type(self, obj_type):
        """Sets object properties based on its type."""
        self.obj_type = obj_type
        if self.obj_type == "chip":
            self.score_value = CHIP_SCORE
            self.damage_value = 0
            self.image = chip_img
        elif self.obj_type == "rock":
            self.score_value = 0
            self.damage_value = 1
            self.image = rock_img
        elif self.obj_type == "heart":
            self.score_value = 0
            self.damage_value = -1 # Negative damage = heal
            self.image = life_icon_img
        elif self.obj_type == "slowmo":
            self.score_value = 0
            self.damage_value = 0
            self.image = slowmo_img
        else:
            self.score_value = 0
            self.damage_value = 0
            
        # Update dimensions
        if self.image is not None:
            self.height, self.width = self.image.shape[:2]
        else:
            self.width, self.height = self.radius * 2, self.radius * 2

    def fall(self, speed_modifier=1.0):
        """Updates the object's vertical position based on its speed and modifier."""
        if self.active:
            self.y += self.speed * speed_modifier

    def draw(self, frame):
        """Draws the object on the given frame."""
        if not self.active:
            return

        dx = int(self.x - self.width // 2)
        dy = int(self.y - self.height // 2)

        if self.image is not None:
            overlay_image_alpha(frame, self.image, dx, dy)
        else:
            # Fallback
            color = ROCK_COLOR_FALLBACK
            if self.obj_type == "chip": color = CHIP_COLOR_FALLBACK
            elif self.obj_type == "heart": color = (0, 0, 255)
            elif self.obj_type == "slowmo": color = (255, 0, 0)
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, color, -1)

    def is_off_screen(self, h):
        return (self.y - self.height // 2) > h

    def get_rect(self):
        return (self.x - self.width // 2, self.y - self.height // 2, self.width, self.height)

# --- Camera Initialization ---
def list_available_cameras(max_to_check=5):
    """Lists all available camera indexes found."""
    print("Checking for available cameras...")
    available_cameras = []
    for i in range(max_to_check):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            print(f"  - Camera index {i}: Available")
            available_cameras.append(i)
            cap_test.release()
        else:
            print(f"  - Camera index {i}: Not found or in use.")
    if not available_cameras:
        print("Warning: No cameras found.")
    print("-----------------------------------")


# --- UI Pre-rendering ---
def create_ui_template(width, height, game_h):
    """Creates the static UI template (banners, borders, titles)."""
    
    # Calculate dimensions
    game_area_y_offset = BANNER_TOP_HEIGHT + UI_BORDER_THICKNESS
    
    # Create canvas
    canvas = np.full((height, width, 3), BANNER_BG_COLOR, dtype=np.uint8)

    # Draw main UI border
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1),
                  UI_BORDER_COLOR, UI_BORDER_THICKNESS * 2)
    
    # Draw border around the game area
    cv2.rectangle(canvas, 
                  (UI_BORDER_THICKNESS, BANNER_TOP_HEIGHT), 
                  (width - UI_BORDER_THICKNESS, BANNER_TOP_HEIGHT + game_h + UI_BORDER_THICKNESS),
                  UI_BORDER_COLOR, UI_BORDER_THICKNESS)

    # --- Draw Banner Text (using PIL) ---
    
    # Title
    title_txt = "Catch and Win!"
    title_w, title_h = get_pil_text_size(title_txt, title_font)
    title_pos_x = (width - title_w) // 2
    title_pos_y = (BANNER_TOP_HEIGHT - title_h) // 2
    canvas = draw_pil_text(canvas, title_txt, (title_pos_x, title_pos_y),
                           title_font, TEXT_COLOR_UI_BANNER)

    # Instructions
    instr_txt = f"Open your mouth to eat {POINTS_TO_WIN} chips! Avoid rocks."
    instr_w, instr_h = get_pil_text_size(instr_txt, ui_font)
    instr_pos_x = (width - instr_w) // 2
    instr_pos_y = height - BANNER_BOTTOM_HEIGHT + (BANNER_BOTTOM_HEIGHT - instr_h) // 2 - 5
    canvas = draw_pil_text(canvas, instr_txt, (instr_pos_x, instr_pos_y),
                           ui_font, TEXT_COLOR_UI_BANNER)
                           
    return canvas

# --- Main Game Loop ---
def run_game():
    """Main function to run the Catch and Win game."""
    
    # --- Initialization ---
    
    # List available cameras first
    list_available_cameras()
    
    # --- SET YOUR CAMERA INDEX HERE ---
    CAMERA_INDEX = 0 
    # ---------------------------------
    
    print(f"Attempting to open camera {CAMERA_INDEX} by default...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_CAM_HEIGHT)
    
    actual_cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened() or actual_cam_width == 0:
        print(f"Error: Camera {CAMERA_INDEX} could not be opened.")
        err_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err_img, f"Camera {CAMERA_INDEX} failed to open!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(err_img, "Check connection or index.", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow("Camera Error", err_img)
        cv2.waitKey(0)
        return
    
    print(f"Success: Camera {CAMERA_INDEX} opened at {actual_cam_width}x{actual_cam_height}.")

    # Load all assets (images, fonts, sounds)
    load_game_assets()

    # Calculate UI dimensions
    GAME_AREA_X_OFFSET = UI_BORDER_THICKNESS
    GAME_AREA_Y_OFFSET = BANNER_TOP_HEIGHT + UI_BORDER_THICKNESS
    UI_FRAME_WIDTH = actual_cam_width + 2 * UI_BORDER_THICKNESS
    UI_FRAME_HEIGHT = actual_cam_height + BANNER_TOP_HEIGHT + BANNER_BOTTOM_HEIGHT + 2 * UI_BORDER_THICKNESS

    WINDOW_NAME = 'Catch and Win - Mouth Eater!'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # Set to fullscreen
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Pre-render the static UI
    ui_canvas_template = create_ui_template(UI_FRAME_WIDTH, UI_FRAME_HEIGHT, actual_cam_height)

    # Game State Variables
    game_state = "START" # START, PLAYING, PAUSED, GAME_OVER
    score = 0
    lives = PLAYER_LIVES
    game_won = False
    spawn_counter = 0
    current_combo = 0
    hit_flash_timer = 0
    slowmo_timer = 0
    popup_animation_timer = 0 # For ease-in animation

    # Object Pooling
    falling_objects_pool = [FallingObject(0, 0, 1, "chip", active=False) for _ in range(50)]
    active_objects = []
    
    # Floating Text/Effects Pool
    active_effects = [] # List of [text, (x, y), life]

    # Mouth Tracking Variables
    mouth_is_open_flag = False
    mouth_center_x, mouth_center_y = 0, 0
    current_mouth_catch_radius = 30 # Initial value, will be updated by detection

    # Difficulty settings
    current_fall_speed_min = FALLING_SPEED_MIN_INITIAL
    current_fall_speed_max = FALLING_SPEED_MAX_INITIAL
    current_spawn_rate = OBJECT_SPAWN_RATE_INITIAL
    rock_spawn_chance = ROCK_SPAWN_CHANCE_INITIAL
    
    # Win/Lose Popup dimensions
    POPUP_WIDTH = 800 # Increased width again
    POPUP_HEIGHT_WIN = 520 # Increased to fit all content
    POPUP_HEIGHT_LOSE = 250 # Increased height for consistency
    win_popup_packet_resized = None
    if packet_original_for_popup is not None:
        pop_pack_w = 250
        pop_pack_h = int(pop_pack_w * (packet_original_for_popup.shape[0] / packet_original_for_popup.shape[1]))
        win_popup_packet_resized = cv2.resize(packet_original_for_popup, (pop_pack_w, pop_pack_h))

    # To store the last good frame for pause/game over
    last_game_area_frame = np.zeros((actual_cam_height, actual_cam_width, 3), dtype=np.uint8)

    # --- Main Loop ---
    while True:
        # --- 1. Get User Input ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        
        # --- Input Handling based on State ---
        if game_state == "GAME_OVER":
            # Only check for 'R' if animation is done
            if popup_animation_timer >= POPUP_ANIMATION_FRAMES and (key == ord('r') or key == ord('R')):
                game_state = "START"
        elif game_state == "START":
            # Start logic is now based on gesture, not key press
            pass 
        elif game_state == "PLAYING":
            if key == ord('p') or key == ord('P'):
                game_state = "PAUSED"
        elif game_state == "PAUSED":
            if key == ord('p') or key == ord('P'):
                game_state = "PLAYING"


        # Start with a fresh copy of the static UI
        ui_canvas = ui_canvas_template.copy()
        
        # --- 2. Game State Logic & Drawing ---

        if game_state == "START":
            # --- START SCREEN ---
            # Read camera
            if not cap.isOpened(): break
            success, game_area_frame_raw = cap.read()
            if not success or game_area_frame_raw is None: break
            game_area_frame = cv2.flip(game_area_frame_raw, 1)

            # Process for hands
            game_area_frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(game_area_frame, cv2.COLOR_BGR2RGB)
            results_hands = hands.process(frame_rgb)
            game_area_frame.flags.writeable = True

            thumbs_up_detected = False
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    if is_thumbs_up(hand_landmarks):
                        thumbs_up_detected = True
                    # Optional: Draw landmarks for debugging
                    # mp.solutions.drawing_utils.draw_landmarks(
                    #     game_area_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Place the live camera feed
            ui_canvas[GAME_AREA_Y_OFFSET:GAME_AREA_Y_OFFSET + actual_cam_height,
                      GAME_AREA_X_OFFSET:GAME_AREA_X_OFFSET + actual_cam_width] = game_area_frame
            
            # Draw "Show Thumbs Up" message
            start_txt = "Show a Thumbs Up to Start!"
            start_txt_w, start_txt_h = get_pil_text_size(start_txt, title_font)
            start_txt_x = (UI_FRAME_WIDTH - start_txt_w) // 2
            start_txt_y = (UI_FRAME_HEIGHT - start_txt_h) // 2
            ui_canvas = draw_pil_text(ui_canvas, start_txt, (start_txt_x, start_txt_y),
                                   title_font, TEXT_COLOR_UI_BANNER)
            
            if thumbs_up_detected:
                # Reset game variables
                score = 0
                lives = PLAYER_LIVES
                game_won = False
                current_combo = 0
                hit_flash_timer = 0
                slowmo_timer = 0
                popup_animation_timer = 0 # Reset animation
                active_objects = []
                for obj in falling_objects_pool: obj.active = False
                game_state = "PLAYING"

        elif game_state == "PLAYING":
            # --- MAIN GAMEPLAY ---
            if not cap.isOpened(): break
            success, game_area_frame_raw = cap.read()
            if not success or game_area_frame_raw is None: break
            game_area_frame = cv2.flip(game_area_frame_raw, 1)

            # Apply hit flash
            if hit_flash_timer > 0:
                flash_overlay = np.full(game_area_frame.shape, HIT_FLASH_COLOR, dtype=np.uint8)
                game_area_frame = cv2.addWeighted(game_area_frame, 0.6, flash_overlay, 0.4, 0)
                hit_flash_timer -= 1

            # Process with FaceMesh
            game_area_frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(game_area_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            game_area_frame.flags.writeable = True

            # --- Mouth Tracking ---
            mouth_is_open_flag = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    y_upper = face_landmarks.landmark[13].y * actual_cam_height
                    y_lower = face_landmarks.landmark[14].y * actual_cam_height
                    x_left = face_landmarks.landmark[61].x * actual_cam_width
                    x_right = face_landmarks.landmark[291].x * actual_cam_width
                    
                    mouth_vert = abs(y_lower - y_upper)
                    mouth_horiz = abs(x_right - x_left)

                    if mouth_horiz > 0:
                        mouth_aspect_ratio = mouth_vert / mouth_horiz
                        if mouth_aspect_ratio > MOUTH_AR_THRESHOLD:
                            mouth_is_open_flag = True

                    mouth_center_x = (x_left + x_right) / 2
                    mouth_center_y = (y_upper + y_lower) / 2
                    current_mouth_catch_radius = (mouth_horiz * MOUTH_CATCH_RADIUS_FACTOR)
            
            # Handle Slow-Mo
            current_speed_mod = 1.0
            if slowmo_timer > 0:
                current_speed_mod = SLOWMO_FACTOR
                slowmo_timer -= 1
                
            # Game Logic: Spawning
            spawn_counter += 1
            if spawn_counter >= current_spawn_rate:
                spawn_counter = 0
                
                spawn_roll = random.random()
                obj_type = "chip" 
                if spawn_roll < rock_spawn_chance:
                    obj_type = "rock"
                elif spawn_roll < rock_spawn_chance + POWERUP_HEART_SPAWN_CHANCE:
                    obj_type = "heart"
                elif spawn_roll < rock_spawn_chance + POWERUP_HEART_SPAWN_CHANCE + POWERUP_SLOWMO_SPAWN_CHANCE:
                    obj_type = "slowmo"
                
                new_obj = None
                for obj in falling_objects_pool:
                    if not obj.active:
                        new_obj = obj
                        break
                
                if new_obj:
                    new_obj.set_type(obj_type)
                    obj_x = actual_cam_width // 2
                    if actual_cam_width > new_obj.width:
                        obj_x = random.randint(new_obj.width // 2, actual_cam_width - new_obj.width // 2)
                    
                    new_obj.x = obj_x
                    new_obj.y = 0 - new_obj.height // 2
                    new_obj.speed = random.randint(current_fall_speed_min, current_fall_speed_max)
                    new_obj.active = True
                    active_objects.append(new_obj)
            
            # Game Logic: Update & Collisions
            new_active_objects = []
            for obj in active_objects:
                if not obj.active: continue
                
                obj.fall(current_speed_mod)
                obj_x, obj_y, obj_w, obj_h = obj.get_rect()
                obj_center_x = obj_x + obj_w / 2
                obj_center_y = obj_y + obj_h / 2
                
                collided_mouth = False
                if mouth_is_open_flag and results.multi_face_landmarks:
                    dist_sq = (obj_center_x - mouth_center_x)**2 + (obj_center_y - mouth_center_y)**2
                    if dist_sq < (current_mouth_catch_radius + max(obj_w, obj_h) / 2)**2:
                        collided_mouth = True
                
                if collided_mouth:
                    if obj.obj_type == "chip":
                        current_combo += 1
                        bonus = (current_combo // COMBO_BONUS_THRESHOLD)
                        score += obj.score_value + bonus
                        sounds['crunch'].play()
                        text = f"+{obj.score_value + bonus}" if bonus > 0 else "+1"
                        active_effects.append([text, (int(mouth_center_x), int(mouth_center_y)), FLOATING_TEXT_LIFESPAN])
                    elif obj.obj_type == "rock":
                        lives -= obj.damage_value
                        hit_flash_timer = 5 
                        current_combo = 0
                        sounds['rock_hit'].play()
                    elif obj.obj_type == "heart":
                        lives = min(PLAYER_LIVES, lives - obj.damage_value)
                        sounds['powerup'].play()
                        active_effects.append(["+1 Life!", (int(mouth_center_x), int(mouth_center_y)), FLOATING_TEXT_LIFESPAN])
                    elif obj.obj_type == "slowmo":
                        slowmo_timer = SLOWMO_DURATION_FRAMES
                        sounds['slowmo'].play()
                        active_effects.append(["SLOW-MO!", (int(mouth_center_x), int(mouth_center_y)), FLOATING_TEXT_LIFESPAN])
                    obj.active = False
                    
                elif obj.is_off_screen(actual_cam_height):
                    if obj.obj_type == "chip": # Missed a chip
                        current_combo = 0
                    obj.active = False
                else:
                    new_active_objects.append(obj)
            
            active_objects = new_active_objects

            # --- Drawing ---
            for obj in active_objects:
                obj.draw(game_area_frame)
            
            new_effects = []
            for effect in active_effects:
                text, pos, life = effect
                if life > 0:
                    current_y = pos[1] - (FLOATING_TEXT_LIFESPAN - life)
                    draw_text_with_outline(game_area_frame, text, (pos[0], current_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), TEXT_COLOR_OUTLINE, 2)
                    effect[2] -= 1 
                    new_effects.append(effect)
            active_effects = new_effects
            
            # Store this frame for pause/game over
            last_game_area_frame = game_area_frame.copy() 
            
            # Place the game area frame into the UI canvas
            ui_canvas[GAME_AREA_Y_OFFSET:GAME_AREA_Y_OFFSET + actual_cam_height,
                      GAME_AREA_X_OFFSET:GAME_AREA_X_OFFSET + actual_cam_width] = game_area_frame

            # Draw dynamic UI elements
            score_txt = f"Score: {score}"
            draw_text_with_outline(ui_canvas, score_txt, (GAME_AREA_X_OFFSET + 20, GAME_AREA_Y_OFFSET + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR_GAME, TEXT_COLOR_OUTLINE, 2, 2)
            combo_txt = f"Combo: {current_combo}x"
            draw_text_with_outline(ui_canvas, combo_txt, (GAME_AREA_X_OFFSET + 20, GAME_AREA_Y_OFFSET + 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), TEXT_COLOR_OUTLINE, 2, 2)

            if slowmo_timer > 0:
                slow_txt = f"SLOW: {slowmo_timer // 60 + 1}s"
                tsz = cv2.getTextSize(slow_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                draw_text_with_outline(ui_canvas, slow_txt, 
                                       (GAME_AREA_X_OFFSET + actual_cam_width - tsz[0] - 20, GAME_AREA_Y_OFFSET + 80),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), TEXT_COLOR_OUTLINE, 2, 2)

            heart_offset_x = GAME_AREA_X_OFFSET + actual_cam_width - (PLAYER_LIVES * (HEART_SIZE[0] + 10))
            heart_offset_y = GAME_AREA_Y_OFFSET + 15
            for i in range(lives):
                hx = heart_offset_x + i * (HEART_SIZE[0] + 5)
                overlay_image_alpha(ui_canvas, life_icon_img, hx, heart_offset_y)

            # --- Check for State Change ---
            if lives <= 0:
                game_state = "GAME_OVER"
                game_won = False
                sounds['lose'].play()
                popup_animation_timer = 0 # Start animation
            elif score >= POINTS_TO_WIN:
                game_state = "GAME_OVER"
                game_won = True
                sounds['win'].play()
                popup_animation_timer = 0 # Start animation

        elif game_state == "PAUSED":
            # --- PAUSE SCREEN ---
            success, _ = cap.read() # Keep camera buffer clear
            
            # Place the last known game frame
            ui_canvas[GAME_AREA_Y_OFFSET:GAME_AREA_Y_OFFSET + actual_cam_height,
                      GAME_AREA_X_OFFSET:GAME_AREA_X_OFFSET + actual_cam_width] = last_game_area_frame

            # Draw semi-transparent overlay
            draw_transparent_rectangle(ui_canvas, 
                                       (GAME_AREA_X_OFFSET, GAME_AREA_Y_OFFSET, actual_cam_width, actual_cam_height),
                                       (0,0,0), 0.5)

            # Draw "PAUSED" text (using PIL font)
            pause_txt = "PAUSED"
            pause_txt_w, pause_txt_h = get_pil_text_size(pause_txt, title_font)
            pause_txt_x = (UI_FRAME_WIDTH - pause_txt_w) // 2
            pause_txt_y = (UI_FRAME_HEIGHT - pause_txt_h) // 2
            ui_canvas = draw_pil_text(ui_canvas, pause_txt, (pause_txt_x, pause_txt_y),
                                   title_font, TEXT_COLOR_UI_BANNER)

        elif game_state == "GAME_OVER":
            # --- GAME OVER SCREEN (ANIMATED) ---
            # Place the final game frame
            ui_canvas[GAME_AREA_Y_OFFSET:GAME_AREA_Y_OFFSET + actual_cam_height,
                      GAME_AREA_X_OFFSET:GAME_AREA_X_OFFSET + actual_cam_width] = last_game_area_frame
            
            # Darken the game area
            draw_transparent_rectangle(ui_canvas, 
                                       (GAME_AREA_X_OFFSET, GAME_AREA_Y_OFFSET, actual_cam_width, actual_cam_height),
                                       (0,0,0), 0.7)
            
            # --- Animation Logic ---
            if popup_animation_timer < POPUP_ANIMATION_FRAMES:
                popup_animation_timer += 1
            
            # Ease-out quad interpolation for a smoother feel
            t = popup_animation_timer / POPUP_ANIMATION_FRAMES
            anim_progress = 1 - (1 - t) * (1 - t) # t * t for ease-in, 1 - (1-t)^2 for ease-out
            
            current_popup_h = 0
            current_popup_w = int(POPUP_WIDTH * anim_progress)
            if game_won:
                current_popup_h = int(POPUP_HEIGHT_WIN * anim_progress)
            else:
                current_popup_h = int(POPUP_HEIGHT_LOSE * anim_progress)

            current_popup_x = (UI_FRAME_WIDTH - current_popup_w) // 2
            current_popup_y = (UI_FRAME_HEIGHT - current_popup_h) // 2
            
            # --- Draw Popup Box ---
            if game_won:
                draw_transparent_rectangle(ui_canvas, (current_popup_x, current_popup_y, current_popup_w, current_popup_h),
                                           POPUP_BG_COLOR, POPUP_ALPHA)
                cv2.rectangle(ui_canvas, (current_popup_x, current_popup_y), 
                              (current_popup_x + current_popup_w, current_popup_y + current_popup_h),
                              POPUP_BORDER_WIN, 4)
            else:
                draw_transparent_rectangle(ui_canvas, (current_popup_x, current_popup_y, current_popup_w, current_popup_h),
                                           POPUP_BG_COLOR_LOSE, POPUP_ALPHA)
                cv2.rectangle(ui_canvas, (current_popup_x, current_popup_y), 
                              (current_popup_x + current_popup_w, current_popup_y + current_popup_h),
                              POPUP_BORDER_LOSE, 4)

            # --- Draw Internal Text ONLY if animation is complete ---
            if anim_progress == 1.0:
                if game_won:
                    # "YOU WON!" text
                    win_txt = "YOU WON!"
                    win_txt_w, win_txt_h = get_pil_text_size(win_txt, title_font)
                    win_txt_x = current_popup_x + (current_popup_w - win_txt_w) // 2
                    win_txt_y = current_popup_y + 40
                    ui_canvas = draw_pil_text(ui_canvas, win_txt, (win_txt_x, win_txt_y),
                                           title_font, POPUP_BORDER_WIN)
                    
                    # Packet image
                    pack_y = win_txt_y + win_txt_h + 20
                    if win_popup_packet_resized is not None:
                        pk_x = current_popup_x + (current_popup_w - win_popup_packet_resized.shape[1]) // 2
                        overlay_image_alpha(ui_canvas, win_popup_packet_resized, pk_x, pack_y)
                        reward_y = pack_y + win_popup_packet_resized.shape[0] + 25
                    else:
                        reward_y = pack_y + 80
                    
                    # Reward text (Split into two lines)
                    code_prompt_txt_1 = "Use this code to"
                    code_prompt_txt_2 = "redeem your prize:"
                    
                    code_prompt_1_w, code_prompt_1_h = get_pil_text_size(code_prompt_txt_1, ui_font_small)
                    code_prompt_1_x = current_popup_x + (current_popup_w - code_prompt_1_w) // 2
                    ui_canvas = draw_pil_text(ui_canvas, code_prompt_txt_1, (code_prompt_1_x, reward_y),
                                           ui_font_small, (50, 50, 50))
                    
                    code_prompt_2_w, code_prompt_2_h = get_pil_text_size(code_prompt_txt_2, ui_font_small)
                    code_prompt_2_x = current_popup_x + (current_popup_w - code_prompt_2_w) // 2
                    ui_canvas = draw_pil_text(ui_canvas, code_prompt_txt_2, (code_prompt_2_x, reward_y + code_prompt_1_h + 5),
                                           ui_font_small, (50, 50, 50))

                    # Reward code (Using smaller font to fit)
                    code_txt = REWARD_CODE
                    code_txt_w, code_txt_h = get_pil_text_size(code_txt, ui_font_small) # Changed to ui_font_small
                    code_txt_x = current_popup_x + (current_popup_w - code_txt_w) // 2
                    ui_canvas = draw_pil_text(ui_canvas, code_txt, (code_txt_x, reward_y + code_prompt_1_h + code_prompt_2_h + 15),
                                           ui_font_small, POPUP_BORDER_WIN) # Changed to ui_font_small
                
                else: # Game Lost
                    # "GAME OVER" text
                    msg_txt = "GAME OVER"
                    msg_w, msg_h = get_pil_text_size(msg_txt, title_font)
                    msg_x = current_popup_x + (current_popup_w - msg_w) // 2
                    msg_y = current_popup_y + 70 # Centered vertically more
                    ui_canvas = draw_pil_text(ui_canvas, msg_txt, (msg_x, msg_y),
                                           title_font, POPUP_BORDER_LOSE)
                    
                    # Display final score
                    final_score_txt = f"Final Score: {score}"
                    fs_w, fs_h = get_pil_text_size(final_score_txt, ui_font)
                    fs_x = current_popup_x + (current_popup_w - fs_w) // 2
                    fs_y = msg_y + msg_h + 20
                    ui_canvas = draw_pil_text(ui_canvas, final_score_txt, (fs_x, fs_y),
                                           ui_font, (200, 200, 200))

                # Restart/Quit message (common)
                pr_msg = "Press 'R' to Restart or 'Q' to Quit"
                pr_msg_w, pr_msg_h = get_pil_text_size(pr_msg, ui_font)
                pr_y = current_popup_y + current_popup_h + 50 # Below the popup
                pr_x = (UI_FRAME_WIDTH - pr_msg_w) // 2
                ui_canvas = draw_pil_text(ui_canvas, pr_msg, (pr_x, pr_y),
                                       ui_font, RESTART_QUIT_TEXT_COLOR)


        # --- 3. Display the final frame ---
        cv2.imshow(WINDOW_NAME, ui_canvas)


    # --- Clean up ---
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if face_mesh:
        face_mesh.close()
    if hands:
        hands.close() # Close the hands model
    pygame.mixer.quit()

if __name__ == '__main__':
    run_game()