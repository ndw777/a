# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Taking care of missing data 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1: 3])
X[:, 1: 3] = imputer.transform(X[:, 1: 3])

# Ecoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] =labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set 
from sklearn.cross_validation import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
# Euclidean Distance between P1 and P2 = sqrt((x_2 - x_1)^2 +(y_2 -y_1)^2)) - distance between two points varies by type of quanity
# Ways to Scale Data
# Standardisation = X_stand = (x-mean(X))/(std_devi(x)) - for each observation and each feature withdraw all mean values of the feature and divide it by the standard deviation
# Normalisation = x_norm = (x-min(x)/(max(x)-min(x)) - substract observation feature x by the minimum value of all the feature values divide it by the max(all)- min(all)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Do we need to fit and transform the Dummy Variables bec/ they are 0 and 1
# depends on the context - dummies varibles will not bet lost if not scaled
# after execute the scale places all the variables between -1 and +1 
