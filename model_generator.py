from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def make_model_two_layers(X, output, 
               lr, loss, metrics):

    model = Sequential([
                        Dense(output, input_dim=X.shape[1], activation='relu'),  # First hidden layer
                        Dropout(0.2),                  # Regularization to prevent overfitting
                        Dense(output, activation='relu'),  # Second hidden layer
                        Dense(1)                       # Output layer (regression)
                        ])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss, metrics=metrics)
    return model


def make_model_three_layers(X, output, 
               lr, loss, metrics):

    model = Sequential([
                        Dense(output, input_dim=X.shape[1], activation='relu'),  # First hidden layer
                        Dropout(0.2),                  # Regularization to prevent overfitting
                        Dense(output, activation='relu'),  # Second hidden layer
                        Dropout(0.2),                   # Regularization to prevent overfitting
                        Dense(output, activation='relu'),  # Third hidden layer
                        Dense(1)                       # Output layer (regression)
                        ])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss, metrics=metrics)
    return model


def make_model_four_layers(X, output, 
               lr, loss, metrics):

    model = Sequential([
                        Dense(output, input_dim=X.shape[1], activation='relu'),  # First hidden layer
                        Dropout(0.2),                  # Regularization to prevent overfitting
                        Dense(output, activation='relu'),  # Second hidden layer
                        Dropout(0.2),                   # Regularization to prevent overfitting
                        Dense(output, activation='relu'),  # Third hidden layer
                        Dropout(0.2), 
                        Dense(output, activation='relu'),  # Fourth hidden layer
                        Dense(1)                       # Output layer (regression)
                        ])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss, metrics=metrics)
    return model
