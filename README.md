# Snack ARcade!

---

"Snack ARcade!" is an interactive game built with **OpenCV** and **MediaPipe** that uses your webcam and face detection to create a fun, engaging experience. Open your mouth to "catch" falling chips and earn points, but beware of the falling rocks! The game features a dynamic difficulty system, increasing challenge as you score, and a rewarding end-game screen for winners.

This project is part of a series of "Catch and Win" game designs exploring different interaction methods.

## Features

* **Real-time Face Mesh Detection:** Utilizes MediaPipe's advanced face mesh capabilities to accurately track your mouth.
* **Interactive Gameplay:** "Catch" falling chips by opening your mouth; avoid falling rocks.
* **Dynamic Difficulty:** As your score increases, the game gets progressively faster, and more rocks appear.
* **Score and Lives System:** Keep track of your points and remaining lives.
* **Win/Lose Conditions:** Reach a target score to win a prize code, or lose all your lives to end the game.
* **Customizable Assets:** Easily replace game object images (chips, rocks, life icons, win screen packet) with your own.
* **User-Friendly Interface:** Clear on-screen instructions, score, and lives display.
* **Restart/Quit Options:** Seamlessly restart the game or exit from the end-game screen.
* **Manual Camera Selection:** Choose your desired webcam by index at startup.

---

## Game Designs in this Project

This repository hosts various iterations of the "Catch and Win" game, each exploring a different input mechanism:

### 1. 2D Game (Finger Mapped to Slider)

This design, typically a separate script (or an earlier version of the game), maps finger movements to control a slider at the bottom of the screen. Chips fall from the top, and the player moves the slider left or right with their finger to catch them.

### 2. Chips Game (Hand Tracking)

In this iteration, the game uses MediaPipe's hand tracking to detect the player's hand. Instead of a slider, the player directly uses their hand to intercept falling chips.

### 3. Chips Game (Mouth Tracking - *Current README Focus*)

This is the primary game detailed in this README, where the player uses their **mouth opening** to catch falling chips and avoid rocks, leveraging MediaPipe's face mesh capabilities.

### 4. 3D Prototype (TouchDesigner)

A more experimental and visually rich prototype developed in **TouchDesigner**. This version explores the "Catch and Win" concept in a 3D environment, potentially incorporating more complex visual effects and interactions. You'll find the TouchDesigner project file (`.toe`) within the repository.

---

## Requirements

Before running the game, ensure you have the following installed:

* **Python 3.7+**
* **OpenCV (`opencv-python`)**
* **MediaPipe (`mediapipe`)**
* **NumPy (`numpy`)**

You can install these libraries using pip:

```bash
pip install opencv-python mediapipe numpy
```

For the **3D Prototype**, you will need:

* **TouchDesigner** (download from the official website: [derivative.ca](https://www.derivative.ca/))

---

## Getting Started

1.  **Clone the repository or download the project files.**
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Place Assets (if not already present):**
    Ensure you have the necessary image assets in an `assets` folder at the root of your project directory for the Python games:
    * `assets/win.png` (Chips Packet for win screen)
    * `assets/chip.png` (Falling chip object)
    * `assets/rock.png` (Falling rock object)
    * `assets/life.png` (Life icon)

    If these files are missing, the Python games will use fallback colors/shapes, but the experience will be better with the actual images.

3.  **Run the Mouth Tracking Game:**
    Open your terminal or command prompt, navigate to the project directory, and run the main Python script:

    ```bash
    python your_mouth_game_script_name.py
    ```
    (Replace `your_mouth_game_script_name.py` with the actual name of your Python file for the mouth-tracking game, e.g., `mouth_game.py` or `main.py` if this is your primary script.)

4.  **Select Camera:**
    Upon launching, the game will prompt you to enter a **camera index**.
    * Typically, your primary webcam is **`0`**.
    * If you have multiple cameras (e.g., an integrated webcam and a USB webcam), try `1`, `2`, etc., until you find the correct one.

5.  **Explore Other Designs:**
    * For the **Hand Tracking** or **Finger Slider** games, locate their respective Python scripts in the repository and run them similarly.
    * To explore the **3D Prototype**, open the `.toe` file (TouchDesigner project file) directly using TouchDesigner.

---

## Gameplay (Mouth Tracking Game)

### Objective

* **Catch Chips:** Open your mouth to "eat" the falling chip bags. Each chip caught adds to your score.
* **Avoid Rocks:** Keep your mouth closed when rocks are falling. Catching a rock will reduce your lives.

### Controls

* **Mouth Movement:** Detected automatically via webcam.
* **R (Restart):** Press `R` on the game over or win screen to restart the game.
* **Q (Quit):** Press `Q` at any time to quit the game.

### Scoring and Lives

* You start with **3 lives**.
* The goal is to reach **20 points** to win the game.
* Successfully catching a chip gives you **1 point**.
* Catching a rock reduces your lives by **1**.
* The game ends if you lose all your lives or reach the target score.

---

## Customization

You can easily modify various game parameters and assets in the `Configuration` section at the top of the Python script for the mouth-tracking game:

* **`REQUESTED_CAM_WIDTH` / `REQUESTED_CAM_HEIGHT`**: Adjust the desired camera resolution.
* **UI Dimensions and Colors**: Change the size of banners, border thickness, and all UI colors.
* **Game Parameters**:
    * `PLAYER_LIVES`: Starting number of lives.
    * `POINTS_TO_WIN`: Target score to win.
    * `REWARD_CODE`: The code displayed on the win screen.
    * `FALLING_SPEED_MIN_INITIAL` / `FALLING_SPEED_MAX_INITIAL`: Initial speed range of falling objects.
    * `OBJECT_SPAWN_RATE_INITIAL`: How often objects appear (in frames).
    * `ROCK_SPAWN_CHANCE_INITIAL`: Probability of a rock spawning instead of a chip.
* **Mouth Tracking Parameters**:
    * `MOUTH_AR_THRESHOLD`: Adjust sensitivity for mouth opening detection.
    * `MOUTH_CATCH_RADIUS_FACTOR`: Control the "hitbox" size around your mouth.
* **Asset Paths**: Update paths to your custom images.

---

## Troubleshooting

* **"No working camera found!"**:
    * Ensure your webcam is connected and not being used by another application.
    * Try different camera indices (e.g., `0`, `1`, `2`) when prompted.
    * Check your operating system's privacy settings to ensure applications have permission to access your camera.
    * Restart your computer.
* **Low Frame Rate / Lag**:
    * Reduce `REQUESTED_CAM_WIDTH` and `REQUESTED_CAM_HEIGHT` to a lower resolution (e.g., 640x480).
    * Ensure no other demanding applications are running in the background.
    * Your computer's processing power and camera's capabilities can affect performance.
* **"Asset image not found" warnings**:
    * Make sure the `assets` folder exists in the same directory as your Python script.
    * Verify that the image files (`win.png`, `chip.png`, `rock.png`, `life.png`) are correctly named and located within the `assets` folder.

---

We hope you enjoy exploring the different interaction designs of "Catch and Win"! Which design do you find the most engaging?
