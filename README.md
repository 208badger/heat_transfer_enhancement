# Heat transfer enhancement in pulsating flows. 

Unsteady fluid flows are present in many engineering applications from combustion engines to nuclear reactor cooling systems. Instantaneous heat transfer in unsteady flows is not well understood. Accurate estimations of the unsteady flow heat transfer would play an important role in the design of next generation of engineering applications, the optimisation of parts and targeted waste heat recovery. 

This study used artificial intelligence, machine learning methods to identify the important parameters to unsteady flow heat transfer and estimate the importance of these parameters. The parameters were narrowed down using a correlation matrix and visual comparison. The parameters selected at the pre-model stage were velocity, density, dynamic viscosity, and acceleration. Four machine learning models were trained on these parameters: Linear Regression, Random Forest, XGBoost and an artificial neural network. 
The models were evaluated for accuracy, with the non-linear models, the latter three, all falling within an acceptable level of accuracy, a mean absolute percentage error of 2.5%. XGBoost was the most accurate model with a mean absolute percentage error of 0.38%.

The accuracy of the non-linear models showed that the fluid parameters important to unsteady heat transfer were the four selected at the pre-model stage: velocity, density, dynamic viscosity, and acceleration. The latter three fit well with the current theories of unsteady flow. They validate and emphasise the use of the Womersley number in temporally average heat transfer analysis. However, the fluid acceleration is a new parameter identified by this investigation as important to unsteady flow heat transfer. 

The success of the non-linear models compared to the lack of accuracy of Linear Regression shows that there is a level of non-linearity to the relationship. Further, the differing importance of the parameters gives a second clue to the nature of the relationship. A suggestion of the general form of the relationship between the parameters and the unsteady heat transfer is presented. 

This study was limited to using simulation data. The next stage is to carry out similar investigations experimentally to validate or contradict the findings reported here. The methodological and code framework developed by this investigation provides a base for the future experimental studies.

### Navigating the repo.
The scripts shown in the main folder are the ones cleaned, and commented ready for use and recreation. Provided that all correct packages are installed and folders are created the scripts can be run from the command line using ```py script_name.py```. However, it is recommended that the code be run in sections in a Jupyter notebook. This is to allow for each output to be viewed and checked if it is working correctly. Further, the permutation importance is a manual entry from the user after having run the model scripts. 

A list of the packages required can be found in the ```requirements.txt```. For pip users, they can all be installed by running `py -m pip install -r requirements.txt`.

The scripts should be run in the following order:
1. `data_processing.py`
2. `initial_visualisation.py`
3. Models: `linear_regression.py`, `random_forest.py`, `_xgboost.py`, `neural_network.py`
4. `importance_plotter.py`

The code used during development of the project can be found in the `project development` folder. These are less clean. 

For further questions please email Joe de Souza. 
