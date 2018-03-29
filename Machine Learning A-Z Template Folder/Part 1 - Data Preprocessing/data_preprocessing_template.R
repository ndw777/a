# Data Processing 

# Importing the dataset
dataset = read.csv('Data.csv')


# Splitting the dataset into the Training set and Test set 
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)

# Note: True = train set, False = Test set 
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# a factor in R is no a numeric number 
training_set[, 2:3]= scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])
