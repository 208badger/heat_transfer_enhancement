#  This script contains the process for the training and evaluation of the artificial neural network model.

# Import the modules.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from eli5.sklearn import PermutationImportance
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

# Read the data.
data = pd.read_parquet("./data/all_data.parquet")

# define the target dataframe
y = data.UHTE

# Define the input data. 
x = data[["velocity", "density", "dynamic_viscosity", "acceleration"]]

# Train-test split.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Input scaling.
scaler = MinMaxScaler()

# Fit to the training data.
scaler.fit(x_train)

# Transform the training data.
x_scaled_train = scaler.transform(x_train)

# Transform the test data.
x_scaled_test = scaler.transform(x_test)

# Transform all the data, so predictions can be made for graph plotting later on. 
all_x_scaled = scaler.transform(x)

## Dynamic Neural Network (DNN)
# Build the model using optimised hyperparameters. Details in the appendix of the report. 
def model_builder():
    model = Sequential()
    model.add(Dense(256, input_dim=len(x_train.columns), kernel_initializer = "random_normal", activation = "relu"))
    model.add(Dense(256, kernel_initializer = "random_normal", activation = "relu"))
    model.add(Dense(1, kernel_initializer = "random_normal", activation="relu"))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model 

# Wrap it in a regressor, for evaluation purposes. 
# 'verbose=0' prevents long print out messages when running.
regressor = KerasRegressor(build_fn=model_builder, epochs = 200, batch_size = 30, verbose = 0)

# The validation split is not compulsory. It just gives an extra split to the training data to stop it overfitting. 
regressor.fit(x_scaled_train, y_train, validation_split = 0.2)

# Make predictions.
predictions = regressor.predict(x_scaled_test)

# Calculate the permutation importance. 
perm = PermutationImportance(regressor, random_state=42).fit(x_scaled_test, y_test)
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
plt.savefig(f"./output_graphs/nn_uhte_comparison_{speed}_rpm.png")



