# This script plots the feature importances against one another as seen in the report. 
# It is important to note that the importances have been entered manually.
# The permutation importance was taken from the outputs of the model scripts.

# Import modules
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# feature importance comparison
style.use("seaborn-deep")
bar_width= 0.2
index = np.arange(4)
labels = ["Linear Regression", "Random Forest", "XGBoost", "Neural network"]
lr_perm = [0.75, 0.019, 0.0003, 0.38]
rf_perm = [0.71, 0.27, 0.002, 1.22]
xgb_perm = [0.78, 0.11, 0.17, 1.21]
nn_perm = [4.1, 0.99, 1, 2.4]

vel_perm = [0.75, 0.71, 0.78, 4.1]
dens_perm = [0.019, 0.27, 0.11, 0.99]
dyn_perm = [0.0003, 0.002, 0.17, 1]
acc_perm = [0.38, 1.22, 1.21, 2.4]

fig, ax = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(15)
ax.bar(index, vel_perm, width=0.2, label = "Velocity")
ax.bar(index+bar_width, dens_perm, width=0.2, label = "Density")
ax.bar(index+2*bar_width, dyn_perm, width = 0.2, label="Dynamic viscosity")
ax.bar(index+3*bar_width, acc_perm, width=0.2, label="Acceleration")
ax.set_xticks(index + 1.5*bar_width)
ax.set_xticklabels(labels)
ax.tick_params(axis="x", labelsize=20)
ax.legend(prop={"size":20})
ax.set_ylabel("Permutation importance", size=20)
plt.tight_layout()
plt.savefig("./output_graphs/visualisation/permutation_importance.png")
plt.show()