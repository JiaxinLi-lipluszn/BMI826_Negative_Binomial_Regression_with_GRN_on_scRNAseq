## Import the packages
# General
import pandas as pd
import numpy as np
import re
from scipy import stats
from patsy import dmatrices

# Models
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Others
import arviz as az
import bambi as bmb
import pymc3 as pm

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context

print(f"Running on PyMC3 v{pm.__version__}")

def test():
    print("Imported NB successfully!")

def fitNBRegression(df_train, dependent_var, independent_vars):
    # prameters:
    # df_train is a pandas dataframe that stores the training data
    # dependent_var is the variable to predict
    # independent_vars is a list of regressors
    # Return: 
    # nb regression results in <statsmodels.genmod.generalized_linear_model.GLMResultsWrapper>
    
    # Generate the formular
    expr = dependent_var + " ~ "
    for var in independent_vars:
        expr = expr + " + " + var
        
    # Sperate y_train and X_train
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    # y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    
    # Using the statsmodels GLM class, train the Poisson regression model on the training data set
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit(method="lbfgs")
    # What is the lbfgs methods? Is it resonable to add it here? I'm adding it to avoid NaN or inf weights in the model here
    
    # print out the training summary
    print(poisson_training_results.summary())

    df_train['LAMBDA'] = poisson_training_results.mu

    #add a derived column called 'AUX_OLS_DEP' to the pandas Data Frame. This new column will store the values of the dependent variable of the OLS regression
    df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x[dependent_var] - x['LAMBDA'])**2 - x['LAMBDA']) / x['LAMBDA'], axis=1)

    #use patsy to form the model specification for the OLSR
    ols_expr = """AUX_OLS_DEP ~ LAMBDA - 1"""

    #Configure and fit the OLSR model
    aux_olsr_results = smf.ols(ols_expr, df_train).fit()
    #Print the regression params
    print(aux_olsr_results.params)

    #train the NB2 model on the training data set
    nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit(method="lbfgs")

    #print the training summary
    print(nb2_training_results.summary())

    return nb2_training_results


def predictonNBModel(nb2_training_results, df_test, dependent_var, independent_vars):
    # Parameters:
    # nb2_training_results: a NB regression model
    # df_test: test set with dependent and independent variabels in a pandas dataframe
    # dependent_var is the variable to predict
    # independent_vars is a list of regressors
    # Return:
    # the predicted values 
    
    # Generate the formular
    expr = dependent_var + " ~ "
    for var in independent_vars:
        expr = expr + " + " + var
    
    #make some predictions using our trained NB2 model
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    
    nb2_predictions = nb2_training_results.get_prediction(X_test)

    #print out the predictions
    predictions_summary_frame = nb2_predictions.summary_frame()
    print(predictions_summary_frame)

    #plot the predicted counts versus the actual counts for the test data
#     predicted_counts=predictions_summary_frame['mean']
#     actual_counts = y_test[dependent_var]
#     fig = plt.figure()
#     fig.suptitle('Predicted versus actual')
#     predicted, = plt.plot(X_test.index, predicted_counts, 'go-', label='Predicted counts')
#     actual, = plt.plot(X_test.index, actual_counts, 'ro-', label='Actual counts')
#     plt.legend(handles=[predicted, actual])
#     plt.show()
    
    return predictions_summary_frame

    