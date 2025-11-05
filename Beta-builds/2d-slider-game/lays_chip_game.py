import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import time
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Screen dimensions
CAM_WIDTH, CAM_HEIGHT = 640, 480
GAME_WIDTH, GAME_HEIGHT = 800, 600
SCREEN_WIDTH = CAM_WIDTH + GAME_WIDTH
SCREEN_HEIGHT = max(CAM_HEIGHT, GAME_HEIGHT)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (135, 206, 235)  # Sky blue background
DARK_BLUE = (25, 25, 112)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Game states
STATE_INTRO = 0
STATE_GAME = 1
STATE_OUTRO = 2

# Level thresholds
LEVEL_THRESHOLDS = [100, 200, 300]
MAX_LEVEL = 3

class AssetManager:
    """Manages game assets"""
    def __init__(self):
        self.assets = {}
        self.load_assets()
    
    def load_assets(self):
        """Load all game assets"""
        assets_folder = "assets"
        
        # Try to load images
        asset_files = {
            'chip': 'chip.png',
            'bomb': 'bomb.png',
            'chips_packet': 'chips_packet.png'
        }
        
        for name, filename in asset_files.items():
            filepath = os.path.join(assets_folder, filename)
            try:
                if os.path.exists(filepath):
                    if name == 'chips_packet':
                        # Load chips packet larger for paddle
                        image = pygame.image.load(filepath)
                        self.assets[name] = pygame.transform.scale(image, (120, 70))
                    else:
                        image = pygame.image.load(filepath)
                        self.assets[name] = pygame.transform.scale(image, (40, 40))
                    print(f"Loaded {name} from {filepath}")
                else:
                    print(f"Asset not found: {filepath}, using placeholder")
                    self.assets[name] = self.create_placeholder(name)
            except Exception as e:
                print(f"Error loading {name}: {e}, using placeholder")
                self.assets[name] = self.create_placeholder(name)
    
    def create_placeholder(self, asset_type):
        """Create placeholder graphics if assets not found"""
        if asset_type == 'chips_packet':
            surface = pygame.Surface((120, 80), pygame.SRCALPHA)
            # Create Lay's-style packet
            pygame.draw.rect(surface, YELLOW, (0, 0, 120, 80))
            pygame.draw.rect(surface, RED, (0, 0, 120, 80), 3)
            # Add "LAY'S" text
            font = pygame.font.Font(None, 24)
            text = font.render("LAY'S", True, RED)
            text_rect = text.get_rect(center=(60, 40))
            surface.blit(text, text_rect)
        else:
            surface = pygame.Surface((40, 40), pygame.SRCALPHA)
            if asset_type == 'chip':
                pygame.draw.circle(surface, YELLOW, (20, 20), 18)
                pygame.draw.circle(surface, ORANGE, (20, 20), 18, 3)
            elif asset_type == 'bomb':
                pygame.draw.circle(surface, BLACK, (20, 20), 18)
                pygame.draw.circle(surface, RED, (20, 20), 18, 3)
                # Add fuse
                pygame.draw.line(surface, YELLOW, (20, 2), (25, 8), 3)
        
        return surface
    
    def get_asset(self, name):
        """Get an asset by name"""
        return self.assets.get(name)

class FontManager:
    """Manages arcade-style fonts"""
    def __init__(self):
        self.fonts = {}
        self.load_fonts()
    
    def load_fonts(self):
        """Load arcade-style fonts"""
        # Try to load arcade-style fonts, fallback to system fonts
        font_names = ['Courier New', 'Consolas', 'Monaco', 'Lucida Console']
        
        sizes = [24, 36, 48, 72, 96]
        
        for size in sizes:
            font_loaded = False
            for font_name in font_names:
                try:
                    font = pygame.font.SysFont(font_name, size, bold=True)
                    self.fonts[f'arcade_{size}'] = font
                    font_loaded = True
                    break
                except:
                    continue
            
            if not font_loaded:
                self.fonts[f'arcade_{size}'] = pygame.font.Font(None, size)
    
    def get_font(self, size):
        """Get font by size"""
        return self.fonts.get(f'arcade_{size}', pygame.font.Font(None, size))




class SmoothPaddle:
    """Smooth paddle movement controller"""
    def __init__(self, width, height, paddle_width):
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.target_x = width // 2 - paddle_width // 2
        self.current_x = self.target_x
        self.smoothing_factor = 0.15  # Adjust for more/less smoothing
        self.center_x = width // 2 - paddle_width // 2
    
    def update_target(self, x_pos):
        """Update target position"""
        if x_pos is not None:
            self.target_x = max(0, min(self.width - self.paddle_width, x_pos - self.paddle_width // 2))
    
    def reset_to_center(self):
        """Reset paddle to center position"""
        self.target_x = self.center_x
        self.current_x = self.center_x
    
    def update_smooth(self, dt):
        """Update current position with smooth interpolation"""
        # Smooth interpolation towards target
        diff = self.target_x - self.current_x
        self.current_x += diff * self.smoothing_factor
        
        # Snap to target if very close (prevents infinite tiny movements)
        if abs(diff) < 0.5:
            self.current_x = self.target_x
    
    def get_position(self):
        """Get current paddle position"""
        return int(self.current_x)










class ChipCatchingGame:
    """Main chip catching game with 3 levels"""
    def __init__(self, width, height, asset_manager):
        self.width = width
        self.height = height
        self.asset_manager = asset_manager
        
        # Paddle properties
        self.paddle_width = 120
        self.paddle_height = 60
        self.paddle_y = height - 80
        self.smooth_paddle = SmoothPaddle(width, height, self.paddle_width)
        
        # Game objects
        self.objects = []
        self.spawn_timer = 0
        self.spawn_interval = 1500  # milliseconds
        
        # Game stats
        self.score = 0
        self.lives = 3
        self.level = 1
        self.chips_caught = 0
        self.game_completed = False
        
        # Visual effects
        self.effects = []
    
    def update_paddle(self, x_pos):
        """Update paddle position based on hand position"""
        self.smooth_paddle.update_target(x_pos)
    
    def reset_paddle_to_center(self):
        """Reset paddle to center when hand is re-detected"""
        self.smooth_paddle.reset_to_center()
    
    def update(self, dt):
        """Update game state"""
        # Update smooth paddle movement
        self.smooth_paddle.update_smooth(dt)
        
        # Check level progression
        current_level = 1
        for i, threshold in enumerate(LEVEL_THRESHOLDS):
            if self.score >= threshold:
                current_level = i + 2
        
        # Check if game should end (completed all levels)
        if current_level > MAX_LEVEL:
            self.game_completed = True
            return False
        
        # Update level if changed
        if current_level != self.level:
            old_level = self.level
            self.level = current_level
            self.add_effect(self.width // 2, self.height // 2, f"LEVEL {self.level}!", YELLOW, 3000)
        
        # Adjust difficulty based on level
        base_speed = 120
        level_multiplier = 1 + (self.level - 1) * 0.5
        fall_speed = base_speed * level_multiplier
        
        base_interval = 1800
        self.spawn_interval = max(600, base_interval - (self.level - 1) * 300)
        
        # Spawn new objects
        self.spawn_timer += dt
        if self.spawn_timer > self.spawn_interval:
            # Adjust bomb probability based on level
            bomb_probability = 0.2 + (self.level - 1) * 0.1  # 20% to 40%
            spawn_type = 'bomb' if np.random.random() < bomb_probability else 'chip'
            
            self.objects.append({
                'x': np.random.randint(0, self.width - 40),
                'y': -40,
                'type': spawn_type,
                'speed': fall_speed + np.random.randint(-30, 30)
            })
            self.spawn_timer = 0
        
        # Update objects
        paddle_x = self.smooth_paddle.get_position()
        for obj in self.objects[:]:
            obj['y'] += obj['speed'] * dt / 1000
            
            # Check collision with paddle
            if (obj['y'] + 40 >= self.paddle_y and 
                obj['x'] + 40 >= paddle_x and 
                obj['x'] <= paddle_x + self.paddle_width):
                
                if obj['type'] == 'chip':
                    points = 10 * self.level  # More points for higher levels
                    self.score += points
                    self.chips_caught += 1
                    self.add_effect(obj['x'] + 20, obj['y'] + 20, f"+{points}", GREEN)
                else:  # bomb
                    self.lives -= 1
                    self.add_effect(obj['x'] + 20, obj['y'] + 20, "-1 LIFE", RED)
                
                self.objects.remove(obj)
            
            # Remove objects that fall off screen
            elif obj['y'] > self.height + 40:
                self.objects.remove(obj)
        
        # Update effects
        for effect in self.effects[:]:
            effect['timer'] -= dt
            effect['y'] -= 50 * dt / 1000  # Float upward
            if effect['timer'] <= 0:
                self.effects.remove(effect)
        
        return self.lives > 0 and not self.game_completed
    
    def add_effect(self, x, y, text, color, duration=2000):
        """Add visual effect"""
        self.effects.append({
            'x': x,
            'y': y,
            'text': text,
            'color': color,
            'timer': duration
        })
    
    def draw(self, screen, font_manager):
        """Draw the game"""
        # Blue gradient background
        for y in range(self.height):
            color_ratio = y / self.height
            r = int(135 * (1 - color_ratio * 0.3))
            g = int(206 * (1 - color_ratio * 0.2))
            b = int(235 * (1 - color_ratio * 0.1))
            pygame.draw.line(screen, (r, g, b), (0, y), (self.width, y))
        
        # Draw paddle (chips packet)
        paddle_x = self.smooth_paddle.get_position()
        chips_packet = self.asset_manager.get_asset('chips_packet')
        if chips_packet:
            screen.blit(chips_packet, (paddle_x, self.paddle_y))
        else:
            # Fallback rectangle
            paddle_rect = pygame.Rect(paddle_x, self.paddle_y, self.paddle_width, self.paddle_height)
            pygame.draw.rect(screen, ORANGE, paddle_rect)
            pygame.draw.rect(screen, YELLOW, paddle_rect, 3)
        
        # Draw falling objects
        for obj in self.objects:
            asset = self.asset_manager.get_asset(obj['type'])
            if asset:
                screen.blit(asset, (int(obj['x']), int(obj['y'])))
        
        # Draw effects
        effect_font = font_manager.get_font(24)
        for effect in self.effects:
            alpha = min(255, int(255 * effect['timer'] / 2000))
            text_surface = effect_font.render(effect['text'], True, effect['color'])
            text_surface.set_alpha(alpha)
            screen.blit(text_surface, (effect['x'] - text_surface.get_width() // 2, effect['y']))
        
        # Draw UI
        ui_font = font_manager.get_font(36)
        
        # Score
        score_text = ui_font.render(f"SCORE: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = ui_font.render(f"LIVES: {self.lives}", True, WHITE)
        screen.blit(lives_text, (10, 50))
        
        # Level and progress
        level_text = ui_font.render(f"LEVEL: {self.level}/3", True, WHITE)
        screen.blit(level_text, (self.width - 180, 10))
        
        # Next level progress
        if self.level <= MAX_LEVEL:
            if self.level == 1:
                next_threshold = LEVEL_THRESHOLDS[0]
                progress_text = f"Next: {next_threshold - self.score} pts"
            elif self.level == 2:
                next_threshold = LEVEL_THRESHOLDS[1]
                progress_text = f"Next: {next_threshold - self.score} pts"
            elif self.level == 3:
                next_threshold = LEVEL_THRESHOLDS[2]
                if self.score < next_threshold:
                    progress_text = f"Final: {next_threshold - self.score} pts"
                else:
                    progress_text = "GAME COMPLETE!"
            
            progress_font = font_manager.get_font(24)
            progress_surface = progress_font.render(progress_text, True, YELLOW)
            screen.blit(progress_surface, (self.width - 200, 50))







class GameManager:
    """Manages game states and transitions"""
    def __init__(self):
        self.state = STATE_INTRO
        self.asset_manager = AssetManager()
        self.font_manager = FontManager()
        self.game = None
        self.final_score = 0
        self.intro_timer = 0
        self.outro_timer = 0
        self.game_completed = False
    
    def start_game(self):
        """Start a new game"""
        self.game = ChipCatchingGame(GAME_WIDTH, GAME_HEIGHT, self.asset_manager)
        self.state = STATE_GAME
        self.game_completed = False
    
    def end_game(self):
        """End the current game"""
        if self.game:
            self.final_score = self.game.score
            self.game_completed = self.game.game_completed
        self.state = STATE_OUTRO
        self.outro_timer = 0
    
    def draw_intro(self, screen):
        """Draw intro screen"""
        # Blue gradient background
        for y in range(GAME_HEIGHT):
            color_ratio = y / GAME_HEIGHT
            r = int(25 + (135 - 25) * (1 - color_ratio))
            g = int(25 + (206 - 25) * (1 - color_ratio))
            b = int(112 + (235 - 112) * (1 - color_ratio))
            pygame.draw.line(screen, (r, g, b), (0, y), (GAME_WIDTH, y))
        
        # Title
        title_font = self.font_manager.get_font(72)
        title_text = title_font.render("LAY'S CHIP", True, YELLOW)
        title_rect = title_text.get_rect(center=(GAME_WIDTH // 2, 120))
        screen.blit(title_text, title_rect)
        
        subtitle_text = title_font.render("CATCHER", True, ORANGE)
        subtitle_rect = subtitle_text.get_rect(center=(GAME_WIDTH // 2, 180))
        screen.blit(subtitle_text, subtitle_rect)
        
        # Level info
        level_font = self.font_manager.get_font(36)
        level_info = [
            "3 CHALLENGING LEVELS",
            "Level 1: Reach 100 points",
            "Level 2: Reach 200 points", 
            "Level 3: Reach 300 points"
        ]
        
        start_y = 240
        for i, line in enumerate(level_info):
            color = YELLOW if i == 0 else WHITE
            text = level_font.render(line, True, color)
            text_rect = text.get_rect(center=(GAME_WIDTH // 2, start_y + i * 40))
            screen.blit(text, text_rect)
        
        # Instructions
        inst_font = self.font_manager.get_font(24)
        instructions = [
            "",
            "Move your hand to control the Lay's packet",
            "Catch chips, avoid bombs!",
            "",
            "Show your hand to start"
        ]
        
        start_y = 400
        for i, line in enumerate(instructions):
            if line:
                color = GREEN if "hand" in line.lower() else WHITE
                text = inst_font.render(line, True, color)
                text_rect = text.get_rect(center=(GAME_WIDTH // 2, start_y + i * 25))
                screen.blit(text, text_rect)
    
    def draw_outro(self, screen):
        """Draw outro/game over screen"""
        # Dark blue background with fade effect
        alpha = min(200, self.outro_timer // 10)
        overlay = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        overlay.set_alpha(alpha)
        overlay.fill(DARK_BLUE)
        screen.blit(overlay, (0, 0))
        
        # Title based on completion
        title_font = self.font_manager.get_font(96)
        if self.game_completed:
            title_text = title_font.render("VICTORY!", True, YELLOW)
        else:
            title_text = title_font.render("GAME OVER", True, RED)
        
        title_rect = title_text.get_rect(center=(GAME_WIDTH // 2, 150))
        screen.blit(title_text, title_rect)
        
        # Final score
        score_font = self.font_manager.get_font(48)
        final_score_text = score_font.render(f"FINAL SCORE: {self.final_score}", True, YELLOW)
        score_rect = final_score_text.get_rect(center=(GAME_WIDTH // 2, 250))
        screen.blit(final_score_text, score_rect)
        
        # Performance rating
        if self.game_completed:
            rating = "CHAMPION!"
            rating_color = YELLOW
        elif self.final_score >= 250:
            rating = "AMAZING!"
            rating_color = YELLOW
        elif self.final_score >= 150:
            rating = "GREAT!"
            rating_color = GREEN
        else:
            rating = "GOOD TRY!"
            rating_color = WHITE
        
        rating_font = self.font_manager.get_font(36)
        rating_text = rating_font.render(rating, True, rating_color)
        rating_rect = rating_text.get_rect(center=(GAME_WIDTH // 2, 320))
        screen.blit(rating_text, rating_rect)
        
        # Restart instructions
        restart_font = self.font_manager.get_font(24)
        restart_text = restart_font.render("Press SPACE to play again or ESC to exit", True, WHITE)
        restart_rect = restart_text.get_rect(center=(GAME_WIDTH // 2, 450))
        screen.blit(restart_text, restart_rect)

def detect_pointing_gesture(hand_landmarks):
    """
    Improved finger detection - detects when index finger is prominently extended
    Less restrictive than the original version but still provides gesture feedback
    """
    landmarks = hand_landmarks.landmark
    
    # Get key landmarks
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    
    # Check if index finger is extended (tip is higher than pip joint)
    index_extended = index_tip.y < index_pip.y
    
    # Check if index finger is more extended than middle finger
    index_extension = index_pip.y - index_tip.y
    middle_extension = middle_pip.y - middle_tip.y
    
    # Check if hand is in a reasonable position (not too tilted)
    hand_reasonable = abs(index_tip.y - wrist.y) > 0.1  # Hand not flat
    
    # More lenient detection - just check if index is reasonably extended
    is_pointing = (index_extended and 
                   index_extension > middle_extension * 0.7 and 
                   hand_reasonable)
    
    return is_pointing










def main():
    print("Initializing Lay's Chip Catcher - 3 Level Challenge...")
    
    # Create main window
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Lay's Chip Catcher - 3 Level Challenge")
    
    # Initialize camera
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Please check if your camera is connected and not being used by another application")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    
    print(f"Camera initialized: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Initialize game manager
    game_manager = GameManager()
    game_surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
    
    # Hand tracking variables
    current_hand_x = GAME_WIDTH // 2
    last_hand_update = time.time()
    hand_detected = False
    pointing_detected = False
    was_hand_lost = False
    
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    print("Starting main loop...")
    running = True
    
    while running:
        dt = clock.tick(60)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if game_manager.state == STATE_INTRO:
                        game_manager.start_game()
                    elif game_manager.state == STATE_OUTRO:
                        game_manager.state = STATE_INTRO
        
        # Camera processing
        success, frame = cap.read()
        if success:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = hands.process(frame_rgb)
            
            # Reset detection flags
            prev_hand_detected = hand_detected
            hand_detected = False
            pointing_detected = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Get index finger tip position for control (always works)
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, _ = frame.shape
                    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Check for pointing gesture (for visual feedback)
                    pointing_detected = detect_pointing_gesture(hand_landmarks)
                    
                    # Always use hand position for control, regardless of gesture
                    current_hand_x = int(index_tip.x * GAME_WIDTH)
                    last_hand_update = time.time()
                    hand_detected = True
                    
                    # Visual feedback based on gesture quality
                    if pointing_detected:
                        # Green circle for good pointing gesture
                        cv2.circle(frame, (index_x, index_y), 15, (0, 255, 0), -1)
                        cv2.circle(frame, (index_x, index_y), 25, (255, 255, 0), 3)
                        cv2.putText(frame, "POINTING!", (index_x - 40, index_y - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Blue circle for basic hand detection
                        cv2.circle(frame, (index_x, index_y), 15, (255, 0, 0), -1)
                        cv2.circle(frame, (index_x, index_y), 20, (0, 255, 255), 2)
                        cv2.putText(frame, "HAND", (index_x - 20, index_y - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Draw vertical guide line
                    line_color = (0, 255, 0) if pointing_detected else (255, 0, 0)
                    cv2.line(frame, (index_x, 0), (index_x, h), line_color, 3)
                    
                    # Check if hand was just re-detected after being lost
                    if not prev_hand_detected and was_hand_lost:
                        if game_manager.state == STATE_GAME and game_manager.game:
                            game_manager.game.reset_paddle_to_center()
                        was_hand_lost = False
                    
                    # Auto-start game from intro when hand detected
                    if game_manager.state == STATE_INTRO:
                        game_manager.intro_timer += dt
                        if game_manager.intro_timer > 1000:  # 1 second delay
                            game_manager.start_game()
            
            # Check if hand was lost
            if not hand_detected and prev_hand_detected:
                was_hand_lost = True
            
            # Convert OpenCV frame to Pygame surface
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        else:
            # Create a black surface if camera fails
            frame_surface = pygame.Surface((CAM_WIDTH, CAM_HEIGHT))
            frame_surface.fill(BLACK)
        
        # Update game state
        if game_manager.state == STATE_GAME:
            # Use hand position for control regardless of gesture
            if hand_detected:
                game_manager.game.update_paddle(current_hand_x)
            else:
                # Center paddle when no hand detected (with timeout)
                if (time.time() - last_hand_update) > 2.0:
                    game_manager.game.update_paddle(GAME_WIDTH // 2)
                else:
                    game_manager.game.update_paddle(current_hand_x)
            
            if not game_manager.game.update(dt):
                game_manager.end_game()
        elif game_manager.state == STATE_OUTRO:
            game_manager.outro_timer += dt
        
        # Draw everything
        screen.fill(BLACK)
        
        # Draw camera feed (left side)
        camera_y_offset = (SCREEN_HEIGHT - CAM_HEIGHT) // 2
        screen.blit(frame_surface, (0, camera_y_offset))
        
        # Draw game (right side)
        game_surface.fill(BLUE)
        
        if game_manager.state == STATE_INTRO:
            game_manager.draw_intro(game_surface)
        elif game_manager.state == STATE_GAME:
            game_manager.game.draw(game_surface, game_manager.font_manager)
        elif game_manager.state == STATE_OUTRO:
            if game_manager.game:
                game_manager.game.draw(game_surface, game_manager.font_manager)
            game_manager.draw_outro(game_surface)
        
        screen.blit(game_surface, (CAM_WIDTH, 0))
        
        # Draw UI elements on camera side
        status_text = "Hand detected" if hand_detected else "Show your hand"
        status_color = GREEN if hand_detected else RED
        status_surface = font.render(status_text, True, status_color)
        screen.blit(status_surface, (10, 10))
        
        # Instructions based on game state
        if game_manager.state == STATE_INTRO:
            instructions = [
                "INTRO MODE",
                "Move hand to start game",
                "Or press SPACE"
            ]
        elif game_manager.state == STATE_GAME:
            instructions = [
                "GAME ACTIVE",
                "Move finger left/right to catch chips",
                "Avoid bombs!",
                "Press ESC to exit"
            ]
        else:  # OUTRO
            instructions = [
                "GAME OVER",
                "Press SPACE to restart",
                "Press ESC to exit"
            ]
        
        for i, instruction in enumerate(instructions):
            color = YELLOW if i == 0 else WHITE
            text_surface = font.render(instruction, True, color)
            screen.blit(text_surface, (10, 40 + i * 25))
        
        # Draw divider line
        pygame.draw.line(screen, WHITE, (CAM_WIDTH, 0), (CAM_WIDTH, SCREEN_HEIGHT), 3)
        
        # Update display
        pygame.display.flip()
    
    # Cleanup
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        cv2.destroyAllWindows()