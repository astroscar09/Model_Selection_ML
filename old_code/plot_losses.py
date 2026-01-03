import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import numpy as np

single_pass_4layers = np.loadtxt('losses_single_pass_4_layers.txt')
kfold_4layers = np.loadtxt('losses_per_Epoch_4_Layers.txt')
kfold_3layers = np.loadtxt('losses_per_epoch_3_layers_kfold.txt')

epochs = np.arange(1, 1001, 1)

plt.figure(figsize = (10, 5))
plt.plot(epochs, single_pass_4layers, label='Single Pass 4 Layers', 
         color='blue',
           zorder = 10)

for i in range(1, 6):
    plt.plot(epochs, kfold_4layers[i-1], 
             label=f'KFold 4 Layers Epoch {i}', color='red', 
              zorder = 10)
    
for i in range(1, 6):
    plt.plot(epochs, kfold_3layers[i-1], 
             label=f'KFold 3 Layers Epoch {i}', color='orange', 
              zorder = 10)

#plt.scatter(model_name, rmse_models, label='RMSE', color='red', s = 200, marker = 'o', alpha = 0.8, zorder = 100)

plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Losses', fontsize = 15)
plt.legend(ncols = 3)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid()
plt.yscale('log')
plt.savefig('Losses_vs_Epochs.png')
#plt.show()