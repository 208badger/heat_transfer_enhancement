# This notebook takes care of the initial visualisation of the data. The correlation matrix. 
# It is recommended to run this in sections in a notebook.

# Import modules
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Import the data from the parquet file
data = pd.read_parquet("./data/all_data.parquet")

# Plot and save the correlation matrix
# Drop the columns that are artificial or are non-instantaneous.
data = data.drop(columns=[
    "time", "crank_angle", "rpm"])
data = data.rename(columns={"steady_flow_htc_approx":"dittus-boelter_analogy"})

# Let's plot the correlation of the data set using a heatmap from seaborn library.

corr_data = data.corr()
plt.figure(figsize=(10, 6)) # this is literally just the size of the plot, nothing to do with the dimensions. 
sns.set_style('ticks')
sns.heatmap(corr_data, annot=True)
plt.tight_layout()
plt.savefig("./output_graphs/collinearity.png")

# For plotting the comparison graphs. 
# Most of what is below is just fiddling to ensure the graphs are plotted on one another. 
# The parameters for users to modify are the 'x_axis' to 'units' inputs that edits what the graph plots
# and the titles and labels on the graphs. 

data = pd.read_parquet("./data/all_data.parquet")

x_axis = "crank_angle"
param_1 = "UHTE"
param_2 = "dynamic_viscosity"
label_1 = "UHTE"
label_2 = "Dynamic Viscosity"
units_1 = ""
units_2 = " (kg/ms)"
color_1 = "#1F9168" 
color_2 = "#002060"
x_data = data[x_axis]
label_size = 20
title_size = 30
linewidth = 3
tick_size = 15
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes()

ax2 = ax1.twinx()

lns1 = ax1.plot(x_data, data[param_1], color = color_1, label=label_1, linewidth=linewidth)
lns2 = ax2.plot(x_data, data[param_2], color = color_2, label = label_2,linewidth=linewidth)
# Solution for having two legends
leg = lns1 + lns2
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=0, prop={"size":label_size})

ax1.set_ylabel(label_1, color=color_1, size = label_size)
ax1.set_xlabel("Crank Angle (degrees)", size=label_size)
ax2.set_ylabel(label_2+units_2, color = color_2, size = label_size)
ax1.tick_params(axis = "x", labelsize = tick_size)
ax1.tick_params(axis = "y", labelsize = tick_size)
ax2.tick_params(axis = "y", labelsize = tick_size)

#plt.title(f"{label_1} vs. {label_2}", size=title_size)
plt.savefig(f"./output_graphs/visualisation/{param_1}+{param_2}_on_{x_axis}.png", bbox_inches="tight")

# For plotting distributions of the data.
# Again, only edit the first three variables. 
variable = "dynamic_viscosity"
label = "Dynamic Viscosity"
units = "kg/ms"
hist = sns.displot(data, x=variable, color="#002060")
plt.title(f"{label} Distribution")
plt.xlabel(f"{label} {units}")
plt.tight_layout()
plt.savefig(f"./output_graphs/distributions/{variable}_dist.png")

# Calculate the mean percentage error for a steady state assumption. 
mae_ss = np.average(np.abs(data.UHTE - np.ones(len(data))*100/data.UHTE))
print("Mean average error for steady state", mae_ss)