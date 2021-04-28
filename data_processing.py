# This script takes data created and post processes all the relevant fields to ensure it is ready for the project.
# It is recommended that this script is broken into chunks and run in a notebook.

# Import modules
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import os 
import glob
import re


# Define some functions for post-processing.
def accel_calc(df, rpm):
    # Function that calculates the acceleration from the velocity crank angle plot using engine speed.
    # This applies differently for calculating the fluid acceleration from experiments.

    # Now calculate the acceleration for each. 
    # find negative start point
    val = data.crank_angle[0]

    # Shift the first angle to 0 degrees to then convert to time
    if val < 0:
        data.crank_angle = data.crank_angle.apply(lambda x: x - val)

    # Now use engine speed to convert crank angle to time
    rps = rpm/60
    seconds_per_rev = 1/rps
    data["time"] = data.crank_angle.apply(lambda x: (x/360)*seconds_per_rev)

    # Calculate the gradient of the graph at each point. 
    x = data.time
    y = data.velocity
    f2 = InterpolatedUnivariateSpline(x, y)
    #Get dervative
    der = []
    for i in range(len(y)):

        h = 1e-9
        der.append( ( f2(x[i]+h)-f2(x[i]-h) )/(2*h) )
    der = np.array(der) 

    # Add the acceleration to the dataframe
    data["acceleration"] = der

    # Now convert the crank angle back
    if data.crank_angle[0] > -90:
        data.crank_angle = data.crank_angle.apply(lambda x: x + val)
    return data 

def prandtl_sort(row):
    # This function calculates the Prandtl number.
    prandtl = (row.specific_heat_cap * row.dynamic_viscosity)/row.thermal_cond
    return prandtl

def correl_sort(row):
    # This function calculates the Dittus-Boelter correlation. 
    correl = 0.023 * row.reynolds_no**0.8 * row.prandtl_no**0.3
    # Diameter of pipe is 30mm.  
    h = correl * row.thermal_cond / 0.03
    return h



# Data processing. 

# These are values required to ensure the right engine speed is used when calculated the data.
# This would be done differently experimentally. 
start_rpm = 750
rpm_spacing = 50

# Collect a list of the files in the folder to iterate through.
raw_list = glob.glob("./data/*.csv") # this is wherever the data is saved.

# Process the data
# define empty data frame for all the data. 
all_data = pd.DataFrame(columns=["crank_angle", "total_pressure", "mass_flow_rate", "velocity", "total_temp", "UHTE", "dynamic_viscosity", "kinematic_viscosity", "density", "rpm", "reynolds_no", "specific_heat_cap", "thermal_cond"])

# Iterate through each data set to combine to create one large dataset. 
for csv in raw_list:
    # Find the number in the data name for later use at finding the engine speed. Make sure it doesn't select other numbers in the title.
    s = re.findall(r"\d+", csv)[1]
    s = int(s)
    # Calculate the rpm that the engine as running at for the sample. 
    rpm = start_rpm + s*rpm_spacing

    # read in the data
    data = pd.read_csv(csv)

    # Add logical check to see if the data is about the correct length. Uncomment for printing.
    # print(f"The file is {csv} - this is an engine speed of {rpm}.")
    # if (len(data) > 1600) or len(data) < 500:
    #     print(f"Data is {len(data)} long. Potentially something has gone wrong here.")
    # else:
    #     print(f"Data is {len(data)} long.")

    # rename the columns into something much more comprehensible. This is only required if the data comes from GT. 
    data.rename(columns={
        "Angle01" : "crank_angle",
        "ptot:exhrunner-1:050":"total_pressure",
        "MASSFLW:exhrunner-1:050":"mass_flow_rate",
        "VEL:exhrunner-1:050":"velocity",
        "TTOT:exhrunner-1:050":"total_temp",
        "HG2:exhrunner-1:050":"UHTE",
        "dynvis:exhrunner-1:050":"dynamic_viscosity",
        "kinvis:exhrunner-1:050":"kinematic_viscosity",
        "DENS:exhrunner-1:050":"density",
        "reyno:exhrunner-1:050":"reynolds_no",
        "CP:exhrunner-1:050":"specific_heat_cap",
        "COND:exhrunner-1:050":"thermal_cond",
    }, inplace = True)
    
    # Calculate the acceleration for this data. 
    data = accel_calc(data, rpm)

    # Add the engine speed to the dataframe
    data["rpm"] = np.ones(len(data))*rpm

    data.to_parquet(f"./data/each_speed_parquets/1cy_anechoic_{rpm}.parquet")
    # concatenate data onto all data. 
    all_data = pd.concat([all_data, data], axis=0)

all_data["prandtl_no"] = all_data.apply(lambda x: prandtl_sort(x), axis=1)
all_data["peclet_no"] = all_data.apply(lambda x: x.reynolds_no * x.prandtl_no, axis = 1)
all_data["steady_flow_htc_approx"] = all_data.apply(lambda x: correl_sort(x), axis=1) # this is the Dittus-Boelter correlation. 


# Check for null cells. Just a check that the data processing has gone smoothly. 
null_data = all_data[all_data.isnull().any(axis=1)]
print(len(null_data))

# Save data to parquet. This is much more storage efficient than a csv. Pyarrow or similar will need to be installed as a package though. 
all_data.to_parquet("./data/all_data.parquet")

