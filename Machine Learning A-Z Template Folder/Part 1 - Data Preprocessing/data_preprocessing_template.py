# Data Preprocessing Template

# Importing the libraries (NEED)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset (NEED)
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set - need to evaluate your model on a different set than the set you built your model 
from sklearn.cross_validation import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


# Data Preprocessing Template 
# At the beginning of every machine learning model...
# we will copy all this code and... 
# put it at the beginning of our Machine Learning Models
