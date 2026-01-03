import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

plt.style.use('style.mplstyle')

from sklearn.model_selection import KFold

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

FEATURE_FILE = 'Features_with_Continuum.txt'
PREDICTION_FILE = 'Predictions_with_Continuum.txt'
CHI2_MAX = 100
CHI2_MIN = 0
SN_MIN = 5.3
EW_CUT = 500

features_cols = ['burst',
                 'delayed:age',
                 'delayed:massformed',
                 'delayed:metallicity',
                 'delayed:tau',
                 'dust:Av',
                 'nebular:logU',
                 'stellar_mass',
                 'formed_mass',
                 'sfr',
                 'ssfr',
                 'mass_weighted_age',
                 'mass_weighted_zmet', 
                'redshift']



def read_data(log):

    features = pd.read_csv(FEATURE_FILE, sep = ' ', index_col = 0)

    yval = pd.read_csv(PREDICTION_FILE, sep = ' ', index_col = 0)


    good_chi2_mask = (features.chisq_phot.values < CHI2_MAX) & (features.chisq_phot.values > CHI2_MIN)
    good_sn_mask = yval.sn.values > SN_MIN

    mask = good_chi2_mask & good_sn_mask

    good_data = features[mask]
    good_yvals = yval[mask]

    remove_bad_EW = good_yvals.EW_r < EW_CUT

    good_data = good_data[remove_bad_EW]
    good_yvals = good_yvals[remove_bad_EW]

    feature_df = good_data[features_cols]
    y_val_df = good_yvals[['EW_r']]

    feature_df = good_data[features_cols]

    feature_df['sfr'] = np.log10(feature_df['sfr'].values)

    if log:
        y_val_df = np.log10(good_yvals[['EW_r']])
    else:
        y_val_df = good_yvals[['EW_r']]

    return feature_df, y_val_df


class NN_Model_2_Layers(nn.Module):
    def __init__(self, input_size, hidden_size1):
        
        super(NN_Model_2_Layers, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size1, 1)


    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

def split_data(feature_df, y_val_df):

    X_train, X_test, y_train, y_test = train_test_split(feature_df, 
                                                        y_val_df, 
                                                        test_size=0.2, 
                                                        random_state=423423)
    return X_train, X_test, y_train, y_test



def applying_kfold_validation_NN(model, X_train, y_train, kfolds, epochs, learning_rate = 0.01, 
                                 optimize = 'adam', criteria = 'mse'):

    if criteria == 'mse':
        criterion = nn.MSELoss()
    elif criteria == 'mae':
        criterion = nn.L1Loss()
    elif criteria == 'huber':
        criterion = nn.HuberLoss()

    val_losses = []
    losses_per_epoch = []
    r2_scores = []
    mae_scores = []
    mse_scores = []

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=4232)
    #print(y_train)

    for train_idx, val_idx in kf.split(X_train):
        
        # Split data
        scale_x = StandardScaler()
        scale_y = StandardScaler()

        X_train_scaled = scale_x.fit_transform(X_train.iloc[train_idx])
        X_test_scaled = scale_x.transform(X_train.iloc[val_idx])


        y_train_scaled = y_train.iloc[train_idx].values
        y_test_scaled = y_train.iloc[val_idx].values

        X_train1, X_val = torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train1, y_val = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1), torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 1)

        losses = []

        if optimize == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimize == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimize == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
        elif optimize == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_train1)

            loss = criterion(outputs, y_train1)
            losses.append(loss.detach().numpy())
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
        
        #plt.plot(losses, label = f'Fold {len(val_losses) + 1}')
        losses_per_epoch.append(losses)
        #losses.append(loss.detach().numpy())

        model.eval()

        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            r2_scores.append(r2_score(y_val, val_outputs))
            mae_scores.append(mean_absolute_error(y_val, val_outputs))
            mse_scores.append(mean_squared_error(y_val, val_outputs))
        
        # Store validation loss for this fold
        val_losses.append(val_loss.item())

    pd.DataFrame({'R2': r2_scores, 
                  'MAE': mae_scores, 
                  'MSE': mse_scores}).to_csv('scores_per_epoch_2layers_kfold.csv')

    return val_losses, losses_per_epoch, model, scale_x, criterion

def test_model(model, X_test, y_test, scale_x, criterion):


    X_test_scaled = scale_x.transform(X_test)
    X_test_torch, y_test = torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        
        test_outputs = model(X_test_torch)
        test_loss = criterion(test_outputs, y_test)

    
    #print(f'Test Loss: {test_loss.item()}')
    #print('Model with the K-Fold Cross-Validation')
    print('MAE:', mean_absolute_error(test_outputs, y_test))
    print('MSE:', mean_squared_error(test_outputs, y_test))
    print('R2 Score:', r2_score(test_outputs, y_test))

    return test_outputs, test_loss.item()


def main(model, feature_df, y_val_df, kfolds, epochs):
    
    X_train, X_test, y_train, y_test = split_data(feature_df, y_val_df)

    val_losses, losses_per_epoch, model, scale_x, criterion = applying_kfold_validation_NN(model, 
                                                                                                   X_train,
                                                                                                   y_train, 
                                                                                                   kfolds, 
                                                                                                   epochs, 
                                                                                                   learning_rate = 0.01, 
                                                                                                   optimize = 'adam', 
                                                                                                   criteria = 'mse')
    
    
    predictions, test_loss = test_model(model, X_test, y_test, scale_x, criterion)
    
    return val_losses, losses_per_epoch, y_test, predictions



def print_results(val_losses, losses_per_epoch, r2_scores, mae_scores, mse_scores):
    print("Validation Losses per Fold:")
    for i, loss in enumerate(val_losses):
        print(f"Fold {i + 1}: {loss:.4f}")

    print("\nTraining Losses per Epoch:")
    for i, losses in enumerate(losses_per_epoch):
        print(f"Fold {i + 1}: {losses[-1]:.4f}")

    print("\nR2 Scores per Fold:")
    for i, score in enumerate(r2_scores):
        print(f"Fold {i + 1}: {score:.4f}")

    print("\nMAE Scores per Fold:")
    for i, score in enumerate(mae_scores):
        print(f"Fold {i + 1}: {score:.4f}")

    print("\nMSE Scores per Fold:")
    for i, score in enumerate(mse_scores):
        print(f"Fold {i + 1}: {score:.4f}")

    print("\nAverage Validation Loss: {:.4f}".format(np.mean(val_losses)))
    print("Average R2 Score: {:.4f}".format(np.mean(r2_scores)))
    print("Average MAE Score: {:.4f}".format(np.mean(mae_scores)))
    print("Average MSE Score: {:.4f}".format(np.mean(mse_scores)))


def plot_comparison(y_true, y_test):

    plt.figure(figsize=(10, 5), constrained_layout=True)
    plt.scatter(y_true, y_test, alpha=0.5, s=10)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Test Set Predictions with K-Fold Cross-Validation')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.show()
    #plt.savefig('predictions.png')
    plt.close()

def plot_loss(losses_per_epoch, val_losses):
    
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    for i in range(len(losses_per_epoch)):
        ax0.plot(losses_per_epoch[i], label = f'Fold {i + 1}')
    
    ax0.set_title('Training Loss per Epoch')
    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('Loss')
    ax0.legend()
    
    ax1.plot(val_losses)
    ax1.set_title('Validation Loss per Fold')
    ax1.set_xlabel('Folds')
    ax1.set_ylabel('Loss')

    #plt.show()
    fig.savefig('losses.png')
    plt.close()

if __name__ == "__main__":
    
    # Define parameters
    input_size = len(features_cols)
    hidden_size1 = 132
    #hidden_size2 = 10
    #hidden_size3 = 5
    kfolds = 5
    epochs = 1000
    LOG = False

    feature_df, y_val_df = read_data(LOG)

    #print(y_val_df)

    # Create model
    model = NN_Model_2_Layers(input_size, hidden_size1)

    # Run main function
    val_losses, losses_per_epoch, y_test, y_pred = main(model, feature_df, y_val_df, kfolds, epochs)

    plot_comparison(y_test, y_pred)
    
    #np.savetxt('val_losses_2Layers.txt', val_losses)
    #np.savetxt('losses_per_epoch_2Layers.txt', losses_per_epoch)
    
    #print_results(val_losses, losses_per_epoch, r2_scores, mae_scores, mse_scores)
    #plot_loss(losses_per_epoch, val_losses)