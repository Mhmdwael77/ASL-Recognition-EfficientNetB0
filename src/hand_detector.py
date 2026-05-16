import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

PADDING = 30  # pixels of padding around the hand bounding box


def detect(frame):
    """
    Detect hand landmarks in a frame and return the annotated frame plus a cropped hand ROI.

    Args:
        frame: BGR image from OpenCV.

    Returns:
        annotated_frame: Frame with landmarks drawn on it.
        hand_roi: Cropped BGR image of the hand region, or None if no hand detected.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return frame, None

    hand_landmarks = results.multi_hand_landmarks[0]

    # Draw all 21 landmarks and their connections
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )

    # Compute bounding box from landmark coordinates
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]

    x_min = max(0, int(min(xs)) - PADDING)
    y_min = max(0, int(min(ys)) - PADDING)
    x_max = min(w, int(max(xs)) + PADDING)
    y_max = min(h, int(max(ys)) + PADDING)

    hand_roi = frame[y_min:y_max, x_min:x_max]

    if hand_roi.size == 0:
        return frame, None

    return frame, hand_roi
