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
X_features = pd.DataFrame(df, columns = ['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45'])
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

# Might combine all of them together for plain linear regression...

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
   std_errs = np.sqrt(var_hat) * np.sqrt(diag_vals) # eq 3.12
   return std_errs

# Get Z-scores 
def get_z_scores(beta_hat, std_errs):
  z_scores = np.zeros(len(beta_hat))
  for j in range(len(beta_hat)): # eq 3.12
    z_scores[j] = beta_hat[j]/std_errs[j]
    return z_scores

#1) Linear regression 

beta_hat = train_data(train_x, train_y)
y_hat = test_data(test_x, beta_hat)
mse = get_MSE(test_y, y_hat)

#print("Mean squared error: ", mse)

# Get correlation coefficients

corr_coeffs = np.corrcoef(X_features.T); 

# Table 3.1 with correlations 

table1_data = {' ': ['lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45'],\
     'lcavol': [corr_coeffs[0,1], corr_coeffs[0,2], corr_coeffs[0,3], corr_coeffs[0,4], corr_coeffs[0,5], corr_coeffs[0,6], corr_coeffs[0,7]],\
     'lweight': [' ', corr_coeffs[1,2], corr_coeffs[1,3], corr_coeffs[1,4], corr_coeffs[1,5], corr_coeffs[1,6], corr_coeffs[1,7]],\
     'age': [' ', ' ', corr_coeffs[2,3], corr_coeffs[2,4], corr_coeffs[2,5], corr_coeffs[2,6], corr_coeffs[2,7]],\
     'lbph': [' ', ' ', ' ', corr_coeffs[3,4], corr_coeffs[3,5], corr_coeffs[3,6], corr_coeffs[3,7]],\
     'svi': [' ', ' ', ' ', ' ', corr_coeffs[4,5], corr_coeffs[4,6], corr_coeffs[4,7]],\
     'lcp': [' ', ' ', ' ', ' ', ' ', corr_coeffs[5,6], corr_coeffs[5,7]],\
     'gleason': [' ', ' ', ' ', ' ', ' ', ' ', corr_coeffs[6,7]]\
     }

table1 = pd.DataFrame(data=table1_data)
print(table1)

#Std errors and Z scores

# do we use train or test for this? 
var_hat = est_var_hat(train_y, y_hat, train_x) 
diag_vals = np.diagonal(np.linalg.inv(train_x.T @ train_x))
std_errs = get_std_errors(var_hat, diag_vals)
z_scores = get_z_scores(beta_hat, std_errs) 

# # Table 3.2 with beta_hat, std_errs, z_scores

# table2_data = {'Term': ['intercept', 'lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45'],\
#       'Coefficient': [beta_hat[0], beta_hat[1], beta_hat[2], beta_hat[3], beta_hat[4], beta_hat[5], beta_hat[6], beta_hat[7], beta_hat[8]],\
#       'Std. Error': [std_errs[0], std_errs[1], std_errs[2], std_errs[3], std_errs[4], std_errs[5], std_errs[6], std_errs[7], std_errs[8]],\
#       'Z Score': [z_scores[0], z_scores[1], z_scores[2], z_scores[3], z_scores[4], z_scores[5], z_scores[6], z_scores[7], z_scores[8]]\
#      }

# table2 = pd.DataFrame(data=table2_data)
# table2


# b) Ridge regression. You must also code this one by hand(eq 3.44 to find the betas). 


#ridhge regression shrinks coeffs by imposing penalty on size, penalty on colinearity of events
#ridge coeffs minimize penalized residual sum 

def get_b_ridge(X, lamb, y):
  #I is the pxp identity matrix
  #as lambda increases, b_hat gets smaller
  b_hat_ridge = np.linalg.inv(X.T @ X + lamb* np.identity(X.shape[1])) @ X.T @ y # eq 3.44
  #above func penalizes b_0 in identity, but we get b_0 (y-int) like so
  b_0 = np.mean(y) #pg 64
  return b_0, b_hat_ridge

def get_y_hat_ridge(b_0, b_hat_ridge, X):
  y_hat_ridge = b_0 + X @ b_hat_ridge
  return y_hat_ridge


#Select the optimal value of Lambda by cross-validation using the validation dataset. Report the mean squared error on the test dataset, 
# using the best lambda you found on the validation set. DO NOT USE THE TEST DATASET TO CHOOSE LAMBDA. 
#lambdas start from 1 to 5000, 0 doesn't make sense cuz then it'd just be plain lin reg
lambs = np.linspace(1,5000,10)
error = []
#will sweep lambda to find the optimal value that fits data purrfectly,,, how to vectorize?
for i in lambs:
    b_0, b_hat_ridge = get_b_ridge(train_x, i, train_y)
    y_hat = get_y_hat_ridge(b_0, b_hat_ridge, val_x)
    mse = get_MSE(val_y, y_hat)
    print(mse)
    #error.append(get_MSE(val_y, y_hat))

# min_mse = min(error)
# min_mse_ind = error.index(min_mse)
# op_lamb = lambs[min_mse_ind]

#print(error)

# #Plot a ridge plot similar to figure 3.8, but you can just sweep the lambda parameter (you don't have to scale it to degrees of freedom).
# fig, ax = plt.subplots(1,1)
# ax.plot(np.squeeze(lambs), np.squeeze(b_hat_ridge))
# ax.set_xlabel('Lambda')
# ax.set_ylabel('Coefficients')
# ax.set_title('Ridge Regression Coefficients')
# plt.show()