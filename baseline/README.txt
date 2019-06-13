This directory contains the code for data visualization and baseline code.

**data_visualization.py contains code for visualizing the data. simply run this code will provide the data_visualization graphs shown in our report
**main.py contains the code for baseline model. The model that directly concatenate one-hot vector together.
By setting mode variable in line109 to be 'together' or 'separate', we can using the original 12-class setting or predict age and gender seperately.
For the 12-class setting, we provide the code for logistic regression and gradient boosting (xgboost)
For age predict, we provide code for age prediction
**utils.py this code provide some utility functions for main.py Mainly for load in the data and construct one-hot vector
