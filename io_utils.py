from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

def save_trained_model(model, filename):

    model.save(f"{filename}")

def load_saved_model(filepath):

    loaded_model = load_model(filepath)

    return loaded_model

def save_data(X_train, y_train, X_test, y_test, filepath):
    np.savez(filepath, x_train = X_train, y_train = y_train, 
                       x_test = X_test, y_test = y_test)

    