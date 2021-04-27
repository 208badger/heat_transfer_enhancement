# heat_transfer_enhancement
Unsteady fluid flows are present in many engineering applications from combustion engines to nuclear reactor cooling systems. This is not an area of fluid dynamics that is well understood, especially the instantaneous heat transfer in unsteady flows. Accurate estimations of the unsteady flow heat transfer would play an important role in the design of next generation of engineering applications, the optimisation of parts and targeted waste heat recovery. 

This study used artificial intelligence, machine learning methods to identify the important parameters to unsteady flow heat transfer and estimate the importance of these parameters. The parameters were narrowed down using a correlation matrix and visual comparison before four machine learning models were trained: Linear Regression, Random Forest, XGBoost and an artificial neural network. 

The models were evaluated for accuracy, with the non-linear models, the latter three, all falling within an acceptable level of accuracy, an average percentage error of +-2%. XGBoost was the most accurate model with an accuracy of +-0.17%.

The fluid parameters important to unsteady heat transfer were identified as velocity, density, dynamic viscosity, and acceleration. The latter three fit well with the current theories of unsteady flow, they validate and emphasise the use of the Womersley number in temporally average heat transfer analysis. However, the acceleration is a new parameter identified by this investigation as important to unsteady flow heat transfer. 

This study was limited to using simulation data. The next stage is to carry out similar investigations experimentally to validate or contradict the findings reported here. 
