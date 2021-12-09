## Import the packages
# General
import scipy as sc
import pandas as pd
import numpy as np

def test():
    print("Imported Conditional Gaussian successfully!")

# Select the related variables
def fitConditionalGaussian(df_train, dependent_var, independent_vars):
    # Parameters:
    # df_train is the training matrix with all variables in it, variables as columns.
    # dependent_var is the gene of interest
    # independent_vars is a list of regulators as well as the sequencing depth
    
    # Get some parameters
    k = len(independent_vars)
    
    # Prepare the matrix
    X = df_train[[dependent_var] + independent_vars]
    # Convert to numpy to get the expectation and covariance
    X_T = X.values.T
    means = X_T.mean(axis = 0)
    u_i = means[0]
    u_R = means[1:k+1]
    
    cov = np.cov(X_T)
    Sigma_iR = cov[0,1:k+1]
    Sigma_RR = cov[1:k+1,1:k+1]
    
    regression_coef = Sigma_iR.dot(Sigma_RR)
    return (u_i, u_R, regression_coef, cov)
    
def prediconConditionalGaussian(fitret, df_test, dependent_var, independent_vars):
    u_i, u_R, regression_coef, cov = fitret
    
    k = len(independent_vars)
    # R_test: N * K matrix
    R_test = df_test[independent_vars]
    # R_test: K * N matrix
    R_test = R_test.values.T
    # Convert u_R to K * 1
    u_R = u_R.T.reshape(k, 1)
    x = R_test - u_R
    u_bar = u_i +  regression_coef.dot((R_test - u_R))
    return u_bar
    

    