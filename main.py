import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Organize and scale prostate cancer data 

# Read data in from the txt file
df = pd.read_csv('prostatedata.txt', delimiter = "\t")

# Pad first column with 1's 
df.insert(0, 'intercept', 1)

# Separate into inputs and outputs 
X = pd.DataFrame(df, columns = ['intercept', 'lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45'])
y = pd.DataFrame(df, columns = ['lpsa'])

# Scale features -- standard scaling 
ss = StandardScaler()
data_scaled = ss.fit_transform(X)

# Split data into 80% train, 10% validate, 10% train 
train_x, rest_x, train_y, rest_y = train_test_split(X, y, test_size = 0.2) 
val_x, test_x, val_y, test_y = train_test_split(rest_x, rest_y, test_size = 0.5)

# Convert to Numpy arrays
train_x = train_x.to_numpy() 
test_x = test_x.to_numpy()
train_y = train_y.to_numpy()
test_y = test_y.to_numpy()

#Functions 

# Find betas (weights)
def train_data(X, y):
  beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y # eq 3.6
  return beta_hat

# Use betas to test data output 
def test_data(X, beta_hat): 
  y_hat = X @ beta_hat # eq 3.7
  return y_hat

# Get MSE (mean squared error)
def get_MSE(y, y_hat):
  diff_y = np.square(y-y_hat)
  mse = (np.sum(diff_y))/len(y)
  return mse 

# Estimate var_hat 
def est_var_hat(y, y_hat, X):
  diff_y = np.square(y-y_hat)
  var_hat = (np.sum(diff_y))/(len(y)-len(X[0]) - 1 - 1) # eq 3.8
  return var_hat

# Get standard errors
def get_std_errors(var_hat, diag_vals): 
   std_errs = var_hat * np.sqrt(diag_vals)
   return std_errs

# Get Z-scores 
def get_z_scores(beta_hat, std_errs):
  z_scores = np.zeros(len(beta_hat))
  for j in range(len(beta_hat)): # eq 3.12
    z_scores[j] = beta_hat[j]/std_errs[j]
    return z_scores

#1) Linear regression 

beta_hat = train_data(train_x, train_y)
print(beta_hat)
y_hat = test_data(test_x, beta_hat)
mse = get_MSE(test_y, y_hat)

print("Mean squared error: ", mse)

# Get correlation coefficients

corr_coeffs = np.corrcoef(X_features); 

# note: insert table 3.1 with correlations 

# note: insert table 3.2 with beta_hat, std_errs, z_scores 

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