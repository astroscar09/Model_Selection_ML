import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import numpy as np


rmse_models = [65.52, 5392.7635**(1/2), 6154.985**(1/2), 4572.53**(1/2), 5160**(1/2), 103.45]
mae_models = [47.09, 55.21, np.nan, np.nan]
r2_models  = [0.55, -54.54, -0.084, 0.396, 0.1715]

model_name = ['XGBoost', 'NN 2 Layers', 'NN 3 Layers', 'NN 4 Layers', 'RF', 'DT']

plt.figure(figsize = (10, 5))
plt.scatter(model_name, rmse_models, label='RMSE', color='blue', s = 200, marker = 's', alpha = 0.5, zorder = 10)
plt.scatter(model_name, rmse_models, label='RMSE', color='red', s = 200, marker = 'o', alpha = 0.8, zorder = 100)
plt.xlabel('Models', fontsize = 15)
plt.ylabel('RMSE', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid()
plt.savefig('RMSE_vs_Models.png')
plt.show()