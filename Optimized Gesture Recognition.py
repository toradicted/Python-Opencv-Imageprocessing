import cv2
import mediapipe as mp
import os
import numpy as np

# Constants
GESTURE_IMAGE_PATH = "images"
GESTURE_SIZE = (120, 120)
TIP_IDS = [4, 8, 12, 16, 20]

# Load gesture images
def load_gesture_images():
    gestures = {
        "✌️ Peace": "peace.png",
        "👉 Point": "point.png",
        "✊ Fist": "fist.png",
        "🖐 Open Hand": "open.png",
    }
    return {name: cv2.imread(os.path.join(GESTURE_IMAGE_PATH, file), cv2.IMREAD_UNCHANGED)
            for name, file in gestures.items()}

gesture_images = load_gesture_images()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.6)

# Finger logic
def get_finger_states(landmarks):
    fingers = []
    is_right_hand = landmarks[17].x < landmarks[0].x
    thumb_open = landmarks[4].x < landmarks[3].x if is_right_hand else landmarks[4].x > landmarks[3].x
    fingers.append(thumb_open)
    fingers.extend(landmarks[i].y < landmarks[i - 2].y for i in TIP_IDS[1:])
    return fingers

# Gesture mapping
def recognize_gesture(fingers):
    gestures = {
        (True, False, False, True, True): "✌️ Peace",
        (False, True, False, False, False): "👉 Point",
        (False, False, False, False, False): "✊ Fist",
        (True, True, True, True, True): "🖐 Open Hand"
    }
    return gestures.get(tuple(fingers), None)

# Overlay function
def overlay_image(bg, fg, x, y):
    if fg is None:
        return bg
    fg = cv2.resize(fg, GESTURE_SIZE)
    h, w = fg.shape[:2]

    if y + h > bg.shape[0] or x + w > bg.shape[1]:
        return bg  # Prevent overflow

    if fg.shape[2] == 4:  # With alpha
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (1 - alpha) * bg[y:y+h, x:x+w, c] + alpha * fg[:, :, c]
    else:
        bg[y:y+h, x:x+w] = fg
    return bg

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    gesture = None

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks)
            finger_states = get_finger_states(landmarks.landmark)
            gesture = recognize_gesture(finger_states)

    if gesture and gesture in gesture_images:
        frame = overlay_image(frame, gesture_images[gesture], 10, 10)
        cv2.putText(frame, gesture, (140, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
