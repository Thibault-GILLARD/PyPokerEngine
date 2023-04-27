import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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
input_dim = 14  # Number of input features (use the total number of features you have in your dataset)
model = create_poker_model(input_dim)
model.summary()
model.save('poker_model.h5')

