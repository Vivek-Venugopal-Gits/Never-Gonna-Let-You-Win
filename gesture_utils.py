import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def detect_and_classify_gesture(frame):
    """Detect hand in frame and classify gesture as Rock (0), Paper (1), or Scissors (2)."""
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Get landmarks for the first hand
    landmarks = results.multi_hand_landmarks[0].landmark
    
    # Classify gesture
    return classify_gesture(landmarks)

def classify_gesture(landmarks):
    """Classify gesture based on hand landmarks."""
    # Finger tip and base indices
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    finger_bases = [5, 9, 13, 17]
    
    # Count raised fingers (tip higher than base in y-coordinate)
    raised_fingers = 0
    for tip, base in zip(finger_tips, finger_bases):
        if landmarks[tip].y < landmarks[base].y:
            raised_fingers += 1
    
    # Heuristic classification
    if raised_fingers <= 1:
        return 0  # Rock
    elif raised_fingers >= 4:
        return 1  # Paper
    else:
        return 2  # Scissors