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

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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


FEATURE_FILE = 'Features_with_Continuum.txt'
PREDICTION_FILE = 'Predictions_with_Continuum.txt'
CHI2_MAX = 100
CHI2_MIN = 0
SN_MIN = 5.3
EW_CUT = 500

def read_data(log = False):
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
        y_val_df = np.log10(good_yvals[['EW_r']].values)
    else:
        y_val_df = good_yvals[['EW_r']]

    return feature_df, y_val_df

class NN_Model_4_Layers(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        
        super(NN_Model_4_Layers, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, 1)
        #self.soft = nn.Softplus()

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        #
        #x = F.softmax(x, dim=1)
        #x = self.relu(x)
        #x = self.dropout(x)
        return x
    

plot_labels = {'burst': 'Burstiness',
                'delayed:age': "Age [Gyr]",
                'delayed:massformed': r'Mass Formed [M$_\odot$]',
                'delayed:metallicity': 'Metallicity',
                'delayed:tau': r'$\tau$ [Gyr]',
                'dust:Av': 'Av',
                'nebular:logU': 'logU',
                'stellar_mass': r'Stellar Mass [M$_\odot$]',
                'formed_mass': r'Formed Mass [M$_\odot$]',
                'sfr': 'SFR [M$_\odot$ yr$^{-1}$]',
                'ssfr': 'sSFR [yr$^{-1}$]',
                'mass_weighted_age': 'Mass Weighted Age [Gyr]',
                'mass_weighted_zmet': 'Mass Weighted Metallicity',
                'redshift': 'Redshift'}

def training_model(X_train, y_train):
    
    epochs = 1000
    learning_rate = 0.001

    criterion = nn.MSELoss()

    val_losses = []

    kf = KFold(n_splits=5, shuffle=True, random_state=432)
    plt.figure(figsize = (10, 5), constrained_layout = True)

    model = NN_Model_4_Layers(len(features_cols),
                                128, 64, 32)

    losses_per_epoch = []

    for train_idx, val_idx in kf.split(X_train):

        # Split data
        scale_x = StandardScaler()

        X_train_scaled = scale_x.fit_transform(X_train.iloc[train_idx])
        X_test_scaled = scale_x.transform(X_train.iloc[val_idx])


        Y_train = y_train.iloc[train_idx]
        Y_test = y_train.iloc[val_idx]

        X_train1, X_val = torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train1, y_val = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1), torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)

        losses = []
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_train1)
            
            loss = criterion(y_train1, outputs)
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
            val_loss = criterion(y_val, val_outputs)

        
        # Store validation loss for this fold
        val_losses.append(val_loss.item())
    np.savetxt('losses_per_Epoch_4_Layers.txt', losses_per_epoch)
    return model, val_losses, losses_per_epoch, scale_x, criterion


def train_model_no_kfold(X_train, X_test, y_train, y_test):

    scale = StandardScaler()

    X_train_scaled = scale.fit_transform(X_train)
    X_test_scaled = scale.transform(X_test)


    #y_train_scaled = scale.fit_transform(y_train)
    #y_test_scaled = scale.transform(y_test)

    X_train, y_train = torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train.values)
    X_test, y_test = torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test.values)

    epochs = 1000
    losses_single_pass = []

    model = NN_Model_4_Layers(len(features_cols), 
                                    128, 64, 32)

    #set criterion and optimizer
    #Using the mse loss function
    criterion = nn.MSELoss() #mean squared error
    #criterion = nn.L1Loss() #mean absolute error
    #criterion = nn.SmoothL1Loss() #combines MSE and MAE into one loss function

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #training loop

    for epoch in range(epochs):
        #zero the parameter gradients
        #optimizer.zero_grad()
        
        #forward pass
        y_pred = model.forward(X_train)
        
        
        #calculate loss
        loss = criterion(y_pred, y_train)
        
        losses_single_pass.append(loss.detach().numpy())
        
        #update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    np.savetxt('losses_single_pass_4_layers.txt', losses_single_pass)


    #Testing on the testing set
    with torch.no_grad():
        y_test_pred = model.forward(X_test)
        test_loss = criterion(y_test_pred, y_test)
        
        print(f'Test Loss: {test_loss.item():.4f}')

        print('Model with no K-Fold Cross-Validation')
        print('MAE:', mean_absolute_error(y_test_pred, y_test))
        print('MSE:', mean_squared_error(y_test_pred, y_test))
        print('R2 Score:', r2_score(y_test_pred, y_test))

def plot_loss(losses_per_epoch, val_losses):
    plt.figure(figsize=(10, 5), constrained_layout=True)
    for i, losses in enumerate(losses_per_epoch):
        plt.plot(losses, label=f'Fold {i + 1}')
    plt.plot(val_losses, label='Validation Loss', color='black', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.yscale('log')
    plt.legend()
    plt.show()

def test_model(model, X_test, y_test, scale_x, criterion):


    X_test_scaled = scale_x.transform(X_test)
    X_test_torch, y_test = torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        
        test_outputs = model(X_test_torch)
        test_loss = criterion(test_outputs, y_test)


    print(f'Test Loss: {test_loss.item()}')
    print('Model with the K-Fold Cross-Validation')
    print('MAE:', mean_absolute_error(test_outputs, y_test))
    print('MSE:', mean_squared_error(test_outputs, y_test))
    print('R2 Score:', r2_score(test_outputs, y_test))

    return test_outputs, test_loss.item()

def compare_predictions_plot(y_true, y_test):
    
    plt.figure(figsize=(10, 5), constrained_layout=True)
    plt.scatter(y_true, y_test, alpha=0.5, s=10)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Test Set Predictions with K-Fold Cross-Validation')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label = '1-to-1 Line')
    plt.legend()
    plt.savefig('Predictions_4_layers.png')
    #plt.show()
    #plt.close('all')


def main(X_train, X_test, y_train, y_test):
    
    model, val_losses, losses_per_epoch, scale_x, criterion = training_model(X_train, y_train)
    #plot_loss(losses_per_epoch, val_losses)
    y_pred, loss = test_model(model, X_test, y_test, scale_x, criterion)
    compare_predictions_plot(y_test, y_pred)



if __name__ == "__main__":

    log = False
    print('Logarithmic scale:', log)

    feature_df, y_val_df = read_data()

    X_train, X_test, y_train, y_test = train_test_split(feature_df, 
                                                    y_val_df, 
                                                    test_size=0.2, 
                                                    random_state=42423)
    
    #X_train, X_test, y_train, y_test = read_data(log)
    #main(X_train, X_test, y_train, y_test)

    #main()

    train_model_no_kfold(X_train, X_test, y_train, y_test)