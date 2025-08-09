import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Generate synthetic data with patterns
def generate_synthetic_data(num_sequences=5000, seq_length=5):
    X = []
    y = []
    # Transition probabilities: e.g., Rock (0) -> Paper (1) is more likely
    transition_probs = [
        [0.5, 0.3, 0.2],  # Rock -> [Rock, Paper, Scissors]
        [0.2, 0.5, 0.3],  # Paper -> [Rock, Paper, Scissors]
        [0.3, 0.2, 0.5]   # Scissors -> [Rock, Paper, Scissors]
    ]
    
    for _ in range(num_sequences):
        sequence = [np.random.randint(0, 3)]  # Start with random gesture
        for _ in range(seq_length - 1):
            last_gesture = sequence[-1]
            next_gesture = np.random.choice([0, 1, 2], p=transition_probs[last_gesture])
            sequence.append(next_gesture)
        X.append(sequence)
        # Next gesture follows the same transition probabilities
        y.append(np.random.choice([0, 1, 2], p=transition_probs[sequence[-1]]))
    
    X = np.array(X).reshape(num_sequences, seq_length, 1)
    y = to_categorical(y, num_classes=3)
    return X, y

# Create and train LSTM model
X, y = generate_synthetic_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(64, input_shape=(5, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16, verbose=1)

# Save model
model.save('models/lstm_model.h5')
print("LSTM model saved to models/lstm_model.h5")