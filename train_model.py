def train_model(X_train, y_train, X_test, y_test, model, epochs = 100, batch_size = 32):
    
    print('generating history')
    hist = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,  # Adjust based on performance
                    batch_size=batch_size,
                    verbose=2
                    )
    
    return hist, model