from data_processessing import *
from model_generator import *
from plotting_utils import *
from train_model import *
from evaluate_model import *
from io_utils import *
from pathlib import Path

MODELS_DIR = 'fitted_models'
DATA_DIR = 'train_test_data'
PLOT_DIR = 'plots'

def perform_hyperparameter_grid_search(X_train, y_train, learning_rates = [1e-4, 5e-4, 1e-3], 
                                      hidden_layers = [32, 64, 128]):

    results = []

    for lr in learning_rates:
        for hidden in hidden_layers:

            print(f'Performing fitting on lr: {lr:.2e} and hidden layers: {hidden}')

            model = make_model_three_layers(X_train,
                                    lr=lr,
                                    output=hidden,
                                loss = 'mse', 
                                metrics = ['mae'])

            history = model.fit(
                                    X_train, y_train,
                                    validation_split=0.2,
                                    epochs=100,
                                    batch_size=64,
                                    verbose=0
                                )

            val_loss = min(history.history["val_loss"])

            results.append({
                "lr": lr,
                "Nodes": hidden,
                "val_loss": val_loss
            })

    df = pd.DataFrame(results).sort_values("val_loss")
    print(df.head())


def main(train = 'EW', model_type = 'three_layers', 
         logy = None, 
         test_size = 0.2, random_state = 42, 
         lr = 0.001, loss = 'mse', metrics = ['mae'], nodes = 64):
    

    data_path = Path(f'{DATA_DIR}/data_{model_type}.npz')

    if data_path.exists():
        data = np.load(data_path)
        X_scaled = data['x_train']
        X_test_scaled = data['x_test']

        y_train = data['y_train']
        y_test = data['y_test']

    else:

        X, y_flux, y_EW = read_and_clean_data()

        if train == 'EW':
            y = y_EW
        else:
            y = y_flux

        if logy:
            y = np.log10(y)

        y, outlier_mask = remove_outliers_IQR(y)
        X = X[~outlier_mask]


        X_train, X_test, y_train, y_test = split_data_train_test(X, y, test_size, random_state)

        X_scaled, scaler_x = scale_data(X_train)

        X_test_scaled = scaler_x.transform(X_test)

        save_data(X_scaled, y_train, X_test_scaled, y_test, data_path)

    model_path = Path(f'{MODELS_DIR}/{model_type}.h5')

    if model_path.exists():
        model = load_saved_model(model_path)
    else:

        if model_type == 'two_layers':
            model = make_model_two_layers(X_scaled, output=nodes, lr = lr, loss = loss, metrics=metrics)

        elif model_type == 'three_layers':

            model = make_model_three_layers(X_scaled, output=nodes, lr = lr, loss = loss, metrics=metrics)

        elif model_type == 'four_layers':
            
            model = make_model_four_layers(X_scaled, output=nodes, lr = lr, loss = loss, metrics=metrics)

        hist, model = train_model(X_scaled, y_train, X_test_scaled, y_test, model)
        
        save_trained_model(model, model_path)



    model_evaluation(model, X_test_scaled, y_test)

    y_pred = model_predictions(model, X_test_scaled)
    fig, ax = plot_input_vs_output(y_pred, y_test)
    in_vs_out_path = Path(f'{PLOT_DIR}/Input_vs_Output_{model_type}.pdf')
    fig.savefig(in_vs_out_path)


if __name__ == '__main__':

    main()