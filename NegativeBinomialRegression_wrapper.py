# This is a wrapper script that allow we do the parallel negative regression on the genes in the dataset

import argparse
import os
import sys
import NegativeBinomialRegression as NB

import pandas as pd
import numpy as np

# Parallel working
import sys, datetime
from multiprocessing import Pool


def get_args_parser():
    parser = argparse.ArgumentParser('ToolName', add_help = False)
    
    # Common parameters
    parser.add_argument('--seed', default=42, type=int,
        help="""The random seed number used in the script""")
    parser.add_argument('--workers', default = 10, type = int,
        help="""The number of workers on this task""")
    
    # Input parameters
    parser.add_argument('--df', default=None, type=str,
        help="""The path to the whole dataset (include both train and test)""")
    parser.add_argument('--train', default= True , type=bool,
        help="""If this is training the negative binomial regression model""")
    
    # Output arguments
    parser.add_argument('--outname', default=None, type=str,
        help="""The path to store the output file.""")
    return parser

def wrapper(parameter_list):
    i, gene, df_train, df_test = parameter_list
    dependent_var = gene
    independent_vars = ['log_total_count']
    
    try:
        nb2_training_results = NB.fitNBRegression(df_train, dependent_var, independent_vars)
    except:
        print(f'Can\'t do the NB regression on gene {gene}!')
        ret = None
    else:
        predicted_counts = NB.predictonNBModel(nb2_training_results, df_test, dependent_var, independent_vars)
        predicted_counts[gene] = predicted_counts['mean']
        ret = predicted_counts[gene]
    return (gene, ret)
    


def runNBR(args):
    
    # Process the dataset
    df = pd.read_csv(args.df, index_col = 0)
    ## Set seed
    np.random.seed(args.seed)
    mask = np.random.rand(len(df)) < 0.8

    df_train = df[mask]
    df_test = df[~mask]
    print('Training data set length='+str(len(df_train)))
    print('Testing data set length='+str(len(df_test)))
    
    # Prepare the parameter list for Pool running
    parameter_list = []
    for i, gene in zip(range(df.shape[1]), df.columns):
        parameter_list.append((i, gene, df_train, df_test))
  
    # Pooled run this program
    with Pool(args.workers) as p:
        res = p.map(wrapper, parameter_list)
    
    predictions = pd.DataFrame()
    for r in res:
        gene, predicted_counts = r
        predictions[gene] = predicted_counts
        
    print("Here is the predictions!")
    print(predictions)
    predictions.to_csv(args.outname)                  
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ToolName', parents=[get_args_parser()])
    args = parser.parse_args()
    runNBR(args)