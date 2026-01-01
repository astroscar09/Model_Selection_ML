import numpy as np


def model_predictions(model, X_test):

    y_pred = model.predict(X_test)

    return y_pred

def model_evaluation(model, X_test, y_test):

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {mae:.3f}")

    relative_error = mae / np.mean(y_test)
    print('Relative Error:', relative_error)

