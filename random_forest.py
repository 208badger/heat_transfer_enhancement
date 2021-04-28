#  This script contains the process for the training and evaluation of the Random Forest model.

# Import the modules
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Read in the data. 
data = pd.read_parquet("./data/all_data_dim_num.parquet")

# define the target dataframe
y = data.UHTE

# Define the input data. 
x = data[["velocity", "density", "dynamic_viscosity", "acceleration"]]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Define the model.
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 42, max_depth = 12, max_samples=None)

# Fit to the data. 
regressor.fit(x_train, y_train)

# Print both types of feature importance. 
perm = PermutationImportance(regressor, random_state=42).fit(x_test, y_test)
print("Feature importances using permutation",perm.feature_importances_)

print("Feature importances using MDI ", regressor.feature_importances_)

# Calculate the average percentage error.
predictions = regressor.predict(x_test)
mean_perc_error = np.average((np.abs(y_test-predictions)*100/y_test))
print("Average percentage error ", mean_perc_error)


all_predictions = regressor.predict(x)
data["predictions"] = all_predictions

# select one speed for plotting
speed = 3000
test = data[data.rpm == speed]

# Plot the predictions against UHTE.
plt.plot(test.crank_angle, test.UHTE, label="GT's UHTE", color="#1F9168")
plt.plot(test.crank_angle, test.predictions, label="Model prediction", color="magenta")
plt.legend(loc="best")
plt.xlabel("Crank angle (degrees)")
plt.ylabel("UHTE")
plt.savefig(f"./output_graphs/random_forest_uhte_comparison_{speed}_rpm.png")