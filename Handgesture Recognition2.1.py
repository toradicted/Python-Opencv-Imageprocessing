import cv2
import mediapipe as mp
import os
import numpy as np
from collections import Counter

# ==========================
# CONFIG
# ==========================
GESTURE_IMAGE_PATH = "images"
GESTURE_SIZE = (120, 120)

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]

SMOOTH_FRAMES = 7

# ==========================
# LOAD IMAGES
# ==========================
def load_gesture_images():

    gestures = {
        "Thumbs Up": "thumbs up.png",
        "Metal Sign": "metal sign.png",
        "Call Sign": "call sign.png",
        "Fist / Zero": "fist.png",
        "Point / One": "point.png",
        "Peace / Two": "peace.png",
        "Three": "three.png",
        "Four": "four.png",
        "Open Hand / Five": "open.png"
    }

    loaded = {}

    for name, file in gestures.items():

        path = os.path.join(
            GESTURE_IMAGE_PATH,
            file
        )

        img = cv2.imread(
            path,
            cv2.IMREAD_UNCHANGED
        )

        loaded[name] = img

    return loaded


gesture_images = load_gesture_images()

# ==========================
# MEDIAPIPE
# ==========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,

    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

mp_draw = mp.solutions.drawing_utils

gesture_history = []

# ==========================
# FINGER STATE
# ==========================
def get_finger_states(landmarks):

    fingers = []

    wrist = landmarks[0]

    is_right = landmarks[17].x < wrist.x

    thumb_open = (
        landmarks[4].x < landmarks[3].x
        if is_right
        else landmarks[4].x > landmarks[3].x
    )

    fingers.append(thumb_open)

    for tip in TIP_IDS[1:]:

        fingers.append(
            landmarks[tip].y <
            landmarks[tip - 2].y
        )

    return fingers

# ==========================
# GESTURE RECOGNITION
# ==========================
def recognize_gesture(f):

    mapping = {

        # thumb index middle ring pinky

        (False, True, True, True, True):
            "Thumbs Up",

        (False, False, True, True, False):
            "Metal Sign",

        (False, True, True, True, False):
            "Call Sign",

        (True, True, True, True, True):
            "Fist / Zero",

        (True, False, True, True, True):
            "Point / One",
        
        (True, False, False, True, True):
            "Peace / Two",
        
        (True, False, False, False, True):
            "Three",
        
        (True, False, False, False, False):
            "Four",

        (False, False, False, False, False):
            "Open Hand / Five"
    }

    return mapping.get(
        tuple(f),
        None
    )

# ==========================
# STABILIZER
# ==========================
def smooth_gesture(current):

    global gesture_history

    gesture_history.append(current)

    if len(gesture_history) > SMOOTH_FRAMES:

        gesture_history.pop(0)

    valid = [
        x for x in gesture_history
        if x is not None
    ]

    if len(valid) == 0:

        return None

    return Counter(valid).most_common(1)[0][0]

# ==========================
# OVERLAY
# ==========================
def overlay_image(
    bg,
    fg,
    x,
    y
):

    if fg is None:

        return bg

    fg = cv2.resize(
        fg,
        GESTURE_SIZE
    )

    h, w = fg.shape[:2]

    if y+h > bg.shape[0] or x+w > bg.shape[1]:

        return bg

    if fg.shape[2] == 4:

        alpha = fg[:, :, 3] / 255

        for c in range(3):

            bg[
                y:y+h,
                x:x+w,
                c
            ] = (

                (1-alpha) *
                bg[
                    y:y+h,
                    x:x+w,
                    c
                ]

                +

                alpha *
                fg[:, :, c]

            )

    else:

        bg[
            y:y+h,
            x:x+w
        ] = fg

    return bg

# ==========================
# CAMERA
# ==========================
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:

        break

    frame = cv2.flip(
        frame,
        1
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    gesture = None

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(

                frame,

                hand,

                mp_hands.HAND_CONNECTIONS

            )

            finger_state = get_finger_states(
                hand.landmark
            )

            detected = recognize_gesture(
                finger_state
            )

            gesture = smooth_gesture(
                detected
            )

    if gesture:

        frame = overlay_image(
            frame,
            gesture_images.get(
                gesture
            ),
            10,
            10
        )

        cv2.putText(

            frame,

            gesture,

            (150, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0,255,0),

            2
        )

    cv2.imshow(
        "Gesture Detection",
        frame
    )

    if cv2.waitKey(1) == ord("q"):

        break

cap.release()

cv2.destroyAllWindows()
