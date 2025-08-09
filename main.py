import cv2
import numpy as np
from tensorflow.keras.models import load_model
from gesture_utils import detect_and_classify_gesture
import time

# Initialize variables
gesture_names = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
gesture_history = []
scores = {'User': 0, 'AI': 0, 'Tie': 0}
lstm_model = load_model('models/lstm_model.h5')
last_gesture_time = 0  # Timestamp of last processed gesture
interval_duration = 5  # 5-second interval between rounds

def predict_next(history):
    """Predict next gesture using LSTM model."""
    if len(history) < 5:
        return np.random.randint(0, 3)
    # Prepare input: shape (1, 5, 1)
    input_seq = np.array(history[-5:]).reshape(1, 5, 1)
    pred = lstm_model.predict(input_seq, verbose=0)
    return np.argmax(pred, axis=1)[0]

def ai_counter(pred):
    """Choose AI move to beat predicted user gesture."""
    return (pred + 1) % 3  # Rock (0) -> Paper (1), Paper (1) -> Scissors (2), Scissors (2) -> Rock (0)

def game_result(user, ai):
    """Determine game outcome and update scores."""
    if user == ai:
        scores['Tie'] += 1
        return 'Tie'
    elif (user + 1) % 3 == ai:
        scores['AI'] += 1
        return 'AI Wins'
    scores['User'] += 1
    return 'User Wins'

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables for result display
last_gesture = None
last_pred = None
last_ai_move = None
last_result = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    current_time = time.time()
    
    # Check if 5 seconds have passed since last gesture
    if current_time - last_gesture_time >= interval_duration:
        # Detect and classify gesture
        gesture = detect_and_classify_gesture(frame)
        
        if gesture is not None:
            gesture_history.append(gesture)
            if len(gesture_history) > 5:
                gesture_history.pop(0)
            
            pred = predict_next(gesture_history)
            ai_move = ai_counter(pred)
            result = game_result(gesture, ai_move)
            
            # Update last results
            last_gesture = gesture
            last_pred = pred
            last_ai_move = ai_move
            last_result = result
            last_gesture_time = current_time
    
    # Display results or prompt
    if last_gesture is not None:
        remaining_time = int(interval_duration - (current_time - last_gesture_time))
        if remaining_time < 0:
            remaining_time = 0
        cv2.putText(frame, f'Gesture: {gesture_names[last_gesture]}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Predicted: {gesture_names[last_pred]}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'AI: {gesture_names[last_ai_move]}', (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Result: {last_result}', (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Scores: User={scores["User"]}, AI={scores["AI"]}, Tie={scores["Tie"]}', 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Next round in: {remaining_time}', (10, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(frame, 'No hand detected', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Rock-Paper-Scissors', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()