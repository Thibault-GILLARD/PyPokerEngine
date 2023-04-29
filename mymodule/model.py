import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

"""
Description of the model:
    - 14 features
    - 3 outputs for fold, call, raise
    - 1 output for raise amount
    - 3 hidden layers
    - 128, 64, 32 neurons
    - 0.5 dropout
    - activation function: relu
    - optimizer: adam
    - loss function: categorical_crossentropy, mse
    - metrics: accuracy, mae
 
"""

# Define the model architecture
def create_poker_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    # Separate output layers
    action_output = Dense(3, activation='softmax', name='action_output')(model.output)  # 3 outputs for fold, call, raise
    amount_output = Dense(1, activation='linear', name='amount_output')(model.output)  # 1 output for raise amount

    # Create the final model with multiple outputs
    final_model = tf.keras.Model(inputs=model.inputs, outputs=[action_output, amount_output])

    # Compile the model with appropriate loss functions and metrics
    final_model.compile(optimizer='adam',
                        loss={'action_output': 'categorical_crossentropy', 'amount_output': 'mse'},
                        metrics={'action_output': 'accuracy', 'amount_output': 'mae'})

    return final_model

# Create the model
input_dim = 24  # Number of input features (use the total number of features you have in your dataset)
model = create_poker_model(input_dim)
model.summary()
model.save('poker_model_input24.h5')
print("Model created and saved successfully!")


# description of the model

"""
Description of the imput
    - 14 features
1. Number of players
2. Stack size of each player
3. Your hole cards
4. Community cards (if any)
5. Game state (e.g., pre-flop, flop, turn, river)
6. Opponents' behavior (e.g., betting patterns, frequencies of actions like call, fold, or raise)
7. Hand strength
8. Position relative to the dealer button
9. Pot odds
10. Bet sizing history
11. Player types (if available, e.g., tight-aggressive, loose-passive, etc.)
12. Effective stack sizes
13. Implied odds
14. Fold equity
 
"""

