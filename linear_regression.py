# This script contains the process for the training and evaluation of the linear regression with stochastic gradient descent model. 

# Import the modules 
import numpy as np 
import pandas as pd 
import sklearn.metrics as skm
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from eli5.sklearn import PermutationImportance

# read in the data. 
data = pd.read_parquet("./data/all_data.parquet")

# define the target dataframe
y = data.UHTE

# Define the input data. 
x = data[["velocity", "density", "dynamic_viscosity", "acceleration"]]

# Split into test/train groups
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

# Scale the data. 
scaler = StandardScaler()

# fit scaler and apply transform.
scaler.fit(x_train)

x_scaled = scaler.transform(x_train)

# Also scale the test data.
x_scaled_test = scaler.transform(x_test)

# Scale all the data for use plotting later. 
all_x_scaled = scaler.transform(x)

# Inputting optimal hyperparameters.
regressor = SGDRegressor(learning_rate="constant", eta0=0.001, max_iter=100) # Data is already standardized. 

# Fit the data and make predictions. 
regressor.fit(x_scaled, y_train)
predictions = regressor.predict(x_scaled_test)

# Print the coefficients of the model.
print("Coefficients ",regressor.coef_, regressor.intercept_)

# Calculate the permutation importance. 
perm = PermutationImportance(regressor, random_state=1).fit(x_scaled_test, y_test)
print("Feature importances ",perm.feature_importances_)

# Calculate the average percentage error of the model.
mean_perc_error = np.average((np.abs(y_test-predictions)*100/y_test))
print("Average percentage error ", mean_perc_error)

# Predict on all the data for plotting.
# add the data back as predictions
all_predictions = regressor.predict(all_x_scaled)
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
plt.savefig(f"./output_graphs/regression_uhte_comparison_{speed}_rpm.png")

