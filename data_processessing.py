import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from EDA import *

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


def read_and_clean_data():

    features = pd.read_csv('Features_with_Continuum.txt', sep = ' ', index_col = 0)

    yval = pd.read_csv('Predictions_with_Continuum.txt', sep = ' ', index_col = 0)

    good_chi2_mask = (features.chisq_phot.values < 100) & (features.chisq_phot.values > 0)
    good_sn_mask = yval.sn.values > 5.5

    mask = good_chi2_mask & good_sn_mask

    good_data = features[mask]
    good_yvals = yval[mask]


    remove_bad_EW = good_yvals.EW_r < 1000

    good_data = good_data[remove_bad_EW]
    good_yvals = good_yvals[remove_bad_EW]

    X = good_data[features_cols].values
    y_flux = good_yvals['flux_line'].values
    y_EW = good_yvals['EW_r'].values 


    return X, y_flux, y_EW

def remove_outliers_IQR(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    data_no_outliers = data[~outliers]
    
    return data_no_outliers, outliers


def scale_data(data):

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler

def split_data_train_test(X, y, test_size, random_state):
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=test_size, 
                                                        random_state=random_state
                                                        )
    
    return X_train, X_test, y_train, y_test