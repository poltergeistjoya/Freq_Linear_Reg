import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Organize and scale prostate cancer data 

# read data in from the txt file
df = pd.read_csv('./prostatedata.txt', delimiter = "\t")

# separate into inputs and outputs 
x = pd.DataFrame(df, columns = ['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45'])
y = pd.DataFrame(df, columns = ['lpsa'])

# scale features -- standard scaling 
ss = StandardScaler()
data_scaled = ss.fit_transform(x)

# split data into 80% train, 10% validate, 10% train 
train_x, rest_x, train_y, rest_y = train_test_split(x, y, test_size = 0.2) 
val_x, test_x, val_y, test_y = train_test_split(rest_x, rest_y, test_size = 0.5)

# Replicate the analysis from chapter 3 of this dataset. Divide your data into roughly 80% train, 10% validation, 
# 10% test. You must keep this split for all 3 parts of this assignment in order to compare the methods fairly. 
# Replicate the textbooks analysis of this dataset. by doing the following

# a) Plain old linear regression, with no regularization. You must code this one by hand 
# (i.e use equation 3.6 to find the betas). Report the mean squared error on the test dataset. Replicate tables 3.1 
# and 3.2. You will not need the validation set for this part of the assigment.

# b) Ridge regression. You must also code this one by hand(eq 3.44 to find the betas). Select the optimal value of Lambda 
# by cross-validation using the validation dataset. Report the mean squared error on the test dataset, using the best 
# lambda you found on the validation set. DO NOT USE THE TEST DATASET TO CHOOSE LAMBDA. Plot a ridge plot similar to 
# figure 3.8, but you can just sweep the lambda parameter (you don't have to scale it to degrees of freedom).

# c) Lasso regression: Use the built in packages in sci-kit learn or MATLAB to do a Lasso regression. Select the optimal 
# value of lambda as in part b) and also display a Lasso plot similar to figure 3.10, but again you can just sweep the 
# lambda parameter.

# Next, download a dataset suitable for linear regression from UCI or another repository. For now, this should be a dataset 
# that only has numerical features, with no missing values. Repeat the analysis above on this dataset.

# Which features did the Lasso select for you to include in your model? Do these features make sense? Compute the
#  MSE on the training dataset and the test dataset for all methods and comment on the results. Compare this MSE to a 
# baseline MSE.

# Stretch goal (2 points): Add nonlinear and interaction terms to your dataset and try to improve the performance. 
# Are you able to do so?