import cv2
import mediapipe as mp
import numpy as np
import random
import time

# --- Configuration ---
REQUESTED_CAM_WIDTH = 1280
REQUESTED_CAM_HEIGHT = 720
BANNER_TOP_HEIGHT = 70
BANNER_BOTTOM_HEIGHT = 60
UI_BORDER_THICKNESS = 10
UI_BORDER_COLOR = (50, 50, 50)
BANNER_BG_COLOR = (70, 70, 70)
TEXT_COLOR_UI_BANNER = (220, 220, 220)
TEXT_COLOR_GAME = (255, 255, 255)
HEART_SIZE = (35, 35)
PLAYER_LIVES = 3
POINTS_TO_WIN = 5
REWARD_CODE = "REALCHIPS4U"
FALLING_SPEED_MIN = 3
FALLING_SPEED_MAX = 8
OBJECT_SPAWN_RATE = 45
CHIP_SCORE = 1
CHIP_COLOR_FALLBACK = (0, 220, 220)
ROCK_COLOR_FALLBACK = (100, 100, 100)
PACKET_IMAGE_PATH = 'assets/win.png'
CHIP_IMAGE_PATH = 'assets/chip.png'
ROCK_IMAGE_PATH = 'assets/rock.png'
LIFE_IMAGE_PATH = 'assets/life.png'

# --- Asset Loading ---
packet_img_resized = None
packet_original_for_popup = None
packet_display_width, packet_display_height = 150, 200
chip_img = None
rock_img = None
life_icon_img = None

def load_and_resize_asset(path, target_size=None, asset_name="Asset"):
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: {asset_name} image not found at {path}.")
            return None
        print(f"{asset_name} image loaded from {path}.")
        if target_size:
            return cv2.resize(img, target_size)
        return img
    except Exception as e:
        print(f"Error loading/resizing {asset_name} from {path}: {e}")
        return None

packet_img_orig_load = load_and_resize_asset(PACKET_IMAGE_PATH, asset_name="Chips Packet")
if packet_img_orig_load is not None:
    packet_original_for_popup = packet_img_orig_load.copy()
    aspect_ratio = packet_img_orig_load.shape[0] / packet_img_orig_load.shape[1]
    packet_display_width = 150
    packet_display_height = int(packet_display_width * aspect_ratio)
    packet_img_resized = cv2.resize(packet_img_orig_load, (packet_display_width, packet_display_height))

chip_img = load_and_resize_asset(CHIP_IMAGE_PATH, target_size=(40, 40), asset_name="Chip")
rock_img = load_and_resize_asset(ROCK_IMAGE_PATH, target_size=(45, 45), asset_name="Rock")
life_icon_img = load_and_resize_asset(LIFE_IMAGE_PATH, target_size=HEART_SIZE, asset_name="Life Icon")

# --- Helper Functions ---
def draw_text_with_outline(img, text, pos, font, scale, text_color, outline_color, thickness, outline_thickness=1):
    cv2.putText(img, text, pos, font, scale, outline_color, thickness + outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, text_color, thickness, cv2.LINE_AA)

def overlay_image_alpha(background_img, overlay_img_bgra, x, y):
    if overlay_img_bgra is None: return
    h_overlay, w_overlay = overlay_img_bgra.shape[:2]
    h_background, w_background = background_img.shape[:2]
    x, y = int(x), int(y)
    x1_bg=max(x,0); y1_bg=max(y,0)
    x2_bg=min(x+w_overlay, w_background); y2_bg=min(y+h_overlay, h_background)
    x1_ov=max(0,-x); y1_ov=max(0,-y)
    x2_ov=x1_ov+(x2_bg-x1_bg); y2_ov=y1_ov+(y2_bg-y1_bg)
    if x2_bg<=x1_bg or y2_bg<=y1_bg: return
    roi_background = background_img[y1_bg:y2_bg, x1_bg:x2_bg]
    roi_overlay = overlay_img_bgra[y1_ov:y2_ov, x1_ov:x2_ov]
    if roi_overlay.size==0: return
    if roi_overlay.shape[2]==4:
        alpha = roi_overlay[:,:,3]/255.0; alpha_expanded = np.dstack([alpha]*3)
        try: roi_background[:] = roi_background*(1-alpha_expanded) + roi_overlay[:,:,:3]*alpha_expanded
        except ValueError:
            if roi_background.shape[:2]==alpha_expanded.shape[:2]:
                 roi_background[:] = roi_background*(1-alpha_expanded) + roi_overlay[:,:,:3]*alpha_expanded
    elif roi_overlay.shape[2]==3 and roi_background.shape==roi_overlay.shape: roi_background[:]=roi_overlay

class FallingObject:
    def __init__(self, x, y, speed, obj_type, image=None, radius=20):
        self.x=x; self.y=y; self.speed=speed; self.obj_type=obj_type; self.image=image; self.radius=radius
        if self.image is not None: self.height,self.width = self.image.shape[:2]
        else: self.width,self.height = radius*2,radius*2
    def fall(self): self.y+=self.speed
    def draw(self, frame):
        dx=int(self.x-self.width//2); dy=int(self.y-self.height//2)
        if self.image is not None: overlay_image_alpha(frame, self.image, dx, dy)
        else: cv2.circle(frame, (int(self.x),int(self.y)), self.radius, CHIP_COLOR_FALLBACK if self.obj_type=="chip" else ROCK_COLOR_FALLBACK, -1)
    def is_off_screen(self, h): return (self.y-self.height//2)>h
    def get_rect(self): return (self.x-self.width//2, self.y-self.height//2, self.width, self.height)

def find_working_camera(req_width, req_height, max_to_check=3):
    for i in range(max_to_check):
        print(f"Trying camera index {i}...")
        cap_test = cv2.VideoCapture(i) # You can try cv2.VideoCapture(i, cv2.CAP_DSHOW) on Windows
        if cap_test.isOpened():
            print(f"Camera {i} opened. Setting resolution to {req_width}x{req_height}...")
            cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, req_width)
            cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, req_height)
            
            # Allow some time for settings to apply if needed, though often not required
            # time.sleep(0.1) 

            actual_w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera {i} reports resolution: {actual_w}x{actual_h}")

            ret, frame = cap_test.read()
            if ret and frame is not None:
                if np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 10:
                    print(f"Success: Camera index {i} provides valid frames at {actual_w}x{actual_h}.")
                    # It's important that actual_w and actual_h are used by the rest of the game.
                    return cap_test, actual_w, actual_h
                else:
                    print(f"Info: Camera {i} (at {actual_w}x{actual_h}) gives black/dark screen.")
            else:
                print(f"Info: Camera {i} (at {actual_w}x{actual_h}) failed to read frame after setting resolution.")
            cap_test.release()
        else:
            print(f"Info: Camera index {i} could not be opened.")
    print("Error: No working camera found.")
    return None, 0, 0

def run_game():
    cap, actual_cam_width, actual_cam_height = find_working_camera(REQUESTED_CAM_WIDTH, REQUESTED_CAM_HEIGHT)
    if cap is None:
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "No working camera found!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(error_img, "Please check connections and drivers.", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
        cv2.imshow("Camera Error", error_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    GAME_AREA_X_OFFSET = UI_BORDER_THICKNESS
    GAME_AREA_Y_OFFSET = BANNER_TOP_HEIGHT + UI_BORDER_THICKNESS
    UI_FRAME_WIDTH = actual_cam_width + 2 * UI_BORDER_THICKNESS
    UI_FRAME_HEIGHT = actual_cam_height + BANNER_TOP_HEIGHT + BANNER_BOTTOM_HEIGHT + 2 * UI_BORDER_THICKNESS

    WINDOW_NAME = 'Catch and Win - Chips Challenge!'
    # --- TROUBLESHOOTING STEP: Comment out these two lines if crash persists ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # --- END TROUBLESHOOTING STEP ---


    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.65, min_tracking_confidence=0.6, max_num_hands=1)

    score = 0; lives = PLAYER_LIVES
    game_over = False; game_won = False
    spawn_counter = 0; falling_objects = []
    packet_x, packet_y = actual_cam_width//2 - packet_display_width//2, actual_cam_height - packet_display_height - 20
    grab_thresh_ratio = 0.35

    win_popup_packet_resized = None
    if packet_original_for_popup is not None:
        pop_pack_w = 250
        pop_pack_h = int(pop_pack_w * (packet_original_for_popup.shape[0]/packet_original_for_popup.shape[1]))
        win_popup_packet_resized = cv2.resize(packet_original_for_popup, (pop_pack_w, pop_pack_h))

    while True: # Main loop
        if not cap.isOpened():
            print("Error: Camera disconnected or became unavailable during gameplay.")
            break
        
        success, game_area_frame_raw = cap.read()
        if not success or game_area_frame_raw is None:
            print("Error: Failed to read frame from camera during gameplay.")
            # Potentially try to re-initialize camera or just end game. For now, end.
            # time.sleep(0.5) # Brief pause before breaking
            # cap.release()
            # cap, actual_cam_width, actual_cam_height = find_working_camera(REQUESTED_CAM_WIDTH, REQUESTED_CAM_HEIGHT)
            # if cap is None: break # if re-init fails, break
            # else: continue # if re-init works, try to continue
            break 

        game_area_frame = cv2.flip(game_area_frame_raw, 1)
        frame_rgb = cv2.cvtColor(game_area_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                hx,hy = int(mcp.x*actual_cam_width), int(mcp.y*actual_cam_height)
                packet_x = max(0, min(hx-packet_display_width//2, actual_cam_width-packet_display_width))
                packet_y = max(0, min(hy-packet_display_height//2, actual_cam_height-packet_display_height))

        if not game_over:
            spawn_counter +=1
            if spawn_counter >= OBJECT_SPAWN_RATE:
                spawn_counter=0; obj_type="chip" if random.random()>0.35 else "rock"
                cur_w=40; img_use=None; rad_new=20
                if obj_type=="chip":
                    img_use=chip_img
                    if chip_img is not None: cur_w=chip_img.shape[1]
                    else: cur_w=rad_new*2
                else:
                    img_use=rock_img
                    if rock_img is not None: cur_w=rock_img.shape[1]
                    else: cur_w=rad_new*2
                half_w=cur_w//2; obj_x=actual_cam_width//2
                if actual_cam_width>cur_w: obj_x=random.randint(half_w, actual_cam_width-half_w)
                obj_y=0; speed=random.randint(FALLING_SPEED_MIN, FALLING_SPEED_MAX)
                falling_objects.append(FallingObject(obj_x,obj_y,speed,obj_type,image=img_use,radius=rad_new))

            for i in range(len(falling_objects)-1, -1, -1):
                obj=falling_objects[i]; obj.fall()
                ox,oy,ow,oh = obj.get_rect()
                pgy_s=packet_y; pgy_e=packet_y+int(packet_display_height*grab_thresh_ratio)
                pgx_s=packet_x; pgx_e=packet_x+packet_display_width
                obj_cx=ox+ow/2; obj_by=oy+oh
                collided=False
                if (pgx_s<obj_cx<pgx_e and pgy_s<obj_by<pgy_e+obj.speed):
                    if not (ox+ow<pgx_s or ox>pgx_e or oy+oh<pgy_s or oy>pgy_e): collided=True
                if collided:
                    if obj.obj_type=="chip":
                        score+=CHIP_SCORE
                        if score>=POINTS_TO_WIN: game_won=True; game_over=True
                    else:
                        lives-=1
                        if lives<=0: lives=0; game_over=True
                    del falling_objects[i]
                elif obj.is_off_screen(actual_cam_height): del falling_objects[i]
        
        for obj in falling_objects: obj.draw(game_area_frame)
        overlay_image_alpha(game_area_frame, packet_img_resized, packet_x, packet_y)
        if packet_img_resized is None: cv2.rectangle(game_area_frame, (packet_x,packet_y), (packet_x+packet_display_width,packet_y+packet_display_height),(0,180,0),-1)
        
        ui_canvas = np.full((UI_FRAME_HEIGHT, UI_FRAME_WIDTH, 3), BANNER_BG_COLOR, dtype=np.uint8)
        cv2.rectangle(ui_canvas, (0,0),(UI_FRAME_WIDTH-1,UI_FRAME_HEIGHT-1),UI_BORDER_COLOR,UI_BORDER_THICKNESS*2)
        ui_canvas[GAME_AREA_Y_OFFSET:GAME_AREA_Y_OFFSET+actual_cam_height, GAME_AREA_X_OFFSET:GAME_AREA_X_OFFSET+actual_cam_width] = game_area_frame

        title_txt="Catch and Win!";tsz=cv2.getTextSize(title_txt,cv2.FONT_HERSHEY_TRIPLEX,1.5,3)[0]
        tx,ty=(UI_FRAME_WIDTH-tsz[0])//2, (BANNER_TOP_HEIGHT+tsz[1])//2+UI_BORDER_THICKNESS//2
        draw_text_with_outline(ui_canvas,title_txt,(tx,ty),cv2.FONT_HERSHEY_TRIPLEX,1.5,TEXT_COLOR_UI_BANNER,(20,20,20),3,2)
        instr_txt=f"Win a pack of chips for {POINTS_TO_WIN} chips!";isz=cv2.getTextSize(instr_txt,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)[0]
        ix,iy=(UI_FRAME_WIDTH-isz[0])//2, UI_FRAME_HEIGHT-(BANNER_BOTTOM_HEIGHT+isz[1])//2-UI_BORDER_THICKNESS+10
        draw_text_with_outline(ui_canvas,instr_txt,(ix,iy),cv2.FONT_HERSHEY_SIMPLEX,0.8,TEXT_COLOR_UI_BANNER,(20,20,20),2,1)
        score_txt=f"Score: {score}";spx,spy=GAME_AREA_X_OFFSET+20,GAME_AREA_Y_OFFSET+40
        draw_text_with_outline(ui_canvas,score_txt,(spx,spy),cv2.FONT_HERSHEY_SIMPLEX,1,TEXT_COLOR_GAME,(0,0,0),2)
        hox=GAME_AREA_X_OFFSET+actual_cam_width-(PLAYER_LIVES*(HEART_SIZE[0]+10)); hoy=GAME_AREA_Y_OFFSET+15
        for i in range(lives):
            hx=hox+i*(HEART_SIZE[0]+5)
            if life_icon_img is not None: overlay_image_alpha(ui_canvas,life_icon_img,hx,hoy)
            else: cv2.circle(ui_canvas,(hx+HEART_SIZE[0]//2,hoy+HEART_SIZE[1]//2),HEART_SIZE[0]//2,(0,0,220),-1);cv2.circle(ui_canvas,(hx+HEART_SIZE[0]//2,hoy+HEART_SIZE[1]//2),HEART_SIZE[0]//2,(0,0,150),2)

        if game_over:
            sub_img=ui_canvas[GAME_AREA_Y_OFFSET:GAME_AREA_Y_OFFSET+actual_cam_height, GAME_AREA_X_OFFSET:GAME_AREA_X_OFFSET+actual_cam_width]
            res=cv2.addWeighted(sub_img,0.3,np.full(sub_img.shape,(0,0,0),dtype=np.uint8),0.7,1.0)
            ui_canvas[GAME_AREA_Y_OFFSET:GAME_AREA_Y_OFFSET+actual_cam_height, GAME_AREA_X_OFFSET:GAME_AREA_X_OFFSET+actual_cam_width]=res
            if game_won:
                pop_w,pop_h=500,420; px,py=(UI_FRAME_WIDTH-pop_w)//2,(UI_FRAME_HEIGHT-pop_h)//2
                cv2.rectangle(ui_canvas,(px,py),(px+pop_w,py+pop_h),(200,255,200),-1);cv2.rectangle(ui_canvas,(px,py),(px+pop_w,py+pop_h),(0,150,0),3)
                win_txt="YOU WON!";wtsz=cv2.getTextSize(win_txt,cv2.FONT_HERSHEY_TRIPLEX,2,3)[0]
                draw_text_with_outline(ui_canvas,win_txt,(px+(pop_w-wtsz[0])//2,py+60),cv2.FONT_HERSHEY_TRIPLEX,2,(0,128,0),(255,255,255),3,2)
                pack_by=py+100
                if win_popup_packet_resized is not None:
                    pkx,pky=px+(pop_w-win_popup_packet_resized.shape[1])//2, py+90
                    overlay_image_alpha(ui_canvas,win_popup_packet_resized,pkx,pky); pack_by=pky+win_popup_packet_resized.shape[0]
                else: cv2.putText(ui_canvas,"[CHIPS PACKET]",(px+150,py+180),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2); pack_by=py+220
                cpt1="Use this code to redeem:";cpt1sz=cv2.getTextSize(cpt1,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)[0];cpt1y=pack_by+35
                draw_text_with_outline(ui_canvas,cpt1,(px+(pop_w-cpt1sz[0])//2,cpt1y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(50,50,50),(255,255,255),2)
                cpt2=REWARD_CODE;cpt2sz=cv2.getTextSize(cpt2,cv2.FONT_HERSHEY_TRIPLEX,1.2,2)[0];cpt2y=cpt1y+cpt1sz[1]+20
                draw_text_with_outline(ui_canvas,cpt2,(px+(pop_w-cpt2sz[0])//2,cpt2y),cv2.FONT_HERSHEY_TRIPLEX,1.2,(0,100,0),(255,255,255),2,2)
            else:
                msg="GAME OVER";mcol=(100,100,255);tsz_m,_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_TRIPLEX,2.5,3)
                draw_text_with_outline(ui_canvas,msg,((UI_FRAME_WIDTH-tsz_m[0])//2,(UI_FRAME_HEIGHT-tsz_m[1])//2+20),cv2.FONT_HERSHEY_TRIPLEX,2.5,mcol,(0,0,0),3,2)
            
            pr_msg="Press 'R' to Restart or 'Q' to Quit";psz=cv2.getTextSize(pr_msg,cv2.FONT_HERSHEY_SIMPLEX,0.9,2)[0]
            pr_y=py+pop_h+30 if game_won else UI_FRAME_HEIGHT-BANNER_BOTTOM_HEIGHT-UI_BORDER_THICKNESS-30
            if pr_y+psz[1]>UI_FRAME_HEIGHT-UI_BORDER_THICKNESS-10: pr_y=UI_FRAME_HEIGHT-UI_BORDER_THICKNESS-10-psz[1]
            draw_text_with_outline(ui_canvas,pr_msg,((UI_FRAME_WIDTH-psz[0])//2,pr_y),cv2.FONT_HERSHEY_SIMPLEX,0.9,TEXT_COLOR_UI_BANNER,(0,0,0),2)
            
            cv2.imshow(WINDOW_NAME, ui_canvas)
            key=cv2.waitKey(0) & 0xFF
            if key==ord('r') or key==ord('R'):
                score=0;lives=PLAYER_LIVES;game_over=False;game_won=False;falling_objects=[];spawn_counter=0
                packet_x,packet_y=actual_cam_width//2-packet_display_width//2, actual_cam_height-packet_display_height-20
                continue
            elif key==ord('q') or key==ord('Q'): break
        else:
            cv2.imshow(WINDOW_NAME, ui_canvas)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q') or key==ord('Q'): break
    
    if cap and cap.isOpened(): cap.release() # Ensure cap is released if it was opened
    cv2.destroyAllWindows()
    if 'hands' in locals() and hands: hands.close()

if __name__ == '__main__':
    run_game()