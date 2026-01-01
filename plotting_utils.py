import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13

map_dict = {'burst': 'Burst Fraction',
            'delayed:age': 'Age [Gyr]',
            'delayed:massformed': r'Mass Formed [log(M/$M_{\odot}$)]',
            'delayed:metallicity': r'Metallicity [Z/Z$_{\odot}$]',
            'delayed:tau': 'Tau [Gyr]',
            'dust:Av': 'Av [Mag]',
            'nebular:logU': 'logU',
            'stellar_mass': 'log(M$_{*}$/M$_{\odot}$)',
            'formed_mass': 'Formed Mass',
            'sfr': r'log$_{10}$(SFR) [M$_{\odot}$/yr]',
            'ssfr': 'sSFR [yr$^{-1}$]',
            'mass_weighted_age': 'Mass Weighted Age [Gyr]',
            'mass_weighted_zmet': 'Mass Weighted Zmet [Z/Z$_{\odot}$]'}


def plot_histograms_from_df(df):

    columns = df.columns

    nrows = int(np.ceil(len(columns)/5))
    ncols = 5

    fig, ax = plt.subplots(nrows, ncols, figsize = (15, 8))
    
    ax = ax.flatten()
    
    for i, col in enumerate(columns):
        
        data = df[col].values
        finite_mask = np.isfinite(data)
        data = data[finite_mask]
        min_data, max_data = data.min(), data.max()

        diff = max_data - min_data
        if np.log10(diff) > 2.5:
            data = np.log10(data)

        finite_mask = np.isfinite(data)
        
        ax[i].hist(data[finite_mask], bins = 30, color = 'purple', 
                   edgecolor="black", linewidth=1.5, histtype="bar")
        ax[i].set_xlabel(map_dict.get(col, col))
    plt.tight_layout()
    return fig, ax

def compute_nmad(y_true, y_pred):
    """
    Compute NMAD between true values and predictions.
    """
    delta = y_pred - y_true
    return 1.48 * np.median(np.abs(delta - np.median(delta)))

def plot_input_vs_output(y_pred, y_test):
    
    nmad = compute_nmad(y_test, y_pred)

    fig, ax = plt.subplots()

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--')
    
    ax.scatter(y_test, y_pred, s = 10, color = 'purple')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')

    ax.text(
        0.05, 0.95,
        f"NMAD = {nmad:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.8)
    )
    plt.tight_layout()
    
    return fig, ax