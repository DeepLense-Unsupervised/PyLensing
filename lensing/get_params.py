# Author: Pranath Reddy
# This module is for generating all the different permutations of the parameters

import numpy as np
import pandas as pd
import sys
import itertools
import json

# Decorator function
def decorator_get_params(function):
    
    # Wrapper function
    def wrapper_get_params(*args, **kwargs):
    
        base_params, number_of_samples = function(*args, **kwargs)
    
        params = base_params
        dims = params.shape[0]

        if dims > 21 and params['profile'].values[21] == 'Spherical':
            dims -= 2
            
        params_list = [[None] for _ in range(dims)]
        
        for i in range(dims):
        
            params['distribution'].values[i] = params['distribution'].values[i].lower()

            if params['distribution'].values[i] == 'fixed':
                if len(params['priors'].values[i]) != 1:
                    raise Exception(''' Values have been incorrectly assigned
                    Please Check the priors of the parameter: {}'''.format(params['parameter'].values[i]))
                    sys.exit()
                params_list[i] = params['priors'].values[i]
                
            elif params['distribution'].values[i] == 'uniform':
                try:
                    value_list = np.random.uniform( params['priors'].values[i][0], params['priors'].values[i][1], params['population'].values[i]).tolist()
                    params_list[i] = value_list
                except:
                    raise Exception(''' Values have been incorrectly assigned
                    Please Check the priors and population of the parameter: {}'''.format(params['parameter'].values[i]))
                    sys.exit()
                
        combinations = []
        for combination in list(itertools.product(*params_list)):
            combinations.append(list(combination))
            
        print('Total number of permutations of the parameters is: {}'.format(len(combinations)))
        combinations = np.asarray(combinations)
        np.random.shuffle(combinations)
        
        if dims > 21 and params['profile'].values[21] == 'Spherical':
            col1 = np.full((combinations.shape[0], 1), params['population'].values[23])
            col2 = np.full((combinations.shape[0], 1), params['population'].values[24])
            col3 = np.full((combinations.shape[0], 1), params['priors'].values[23][0])
            col4 = np.full((combinations.shape[0], 1), params['priors'].values[23][1])
            col5 = np.full((combinations.shape[0], 1), params['priors'].values[24][0])
            col6 = np.full((combinations.shape[0], 1), params['priors'].values[24][1])
            combinations = np.concatenate((combinations,col1,col2,col3,col4,col5,col6), axis=1)
        
        if number_of_samples == 'All':
            return combinations
        else:
            if number_of_samples > combinations.shape[0]:
                raise Exception(''' The given number of samples exceeds the total number of permutations
                Please provide a value less than: {}'''.format(combinations.shape[0]))
                sys.exit()
            indices = np.arange(combinations.shape[0])
            np.random.shuffle(indices)
            samples = combinations[indices[:number_of_samples]]
            return samples
        
    return wrapper_get_params

# Function to generate all the permutations for a given set of base parameters
@decorator_get_params
def get_params(base_params, number_of_samples='All'):
    return base_params, number_of_samples
    
# Function to generate all the permutations for a given set of base parameters from a CSV file
@decorator_get_params
def load_from_csv(path, number_of_samples='All'):

    params = pd.read_csv(path)
    dims = params.shape[0]
    
    for i in range(dims):
        params['distribution'].values[i] = params['distribution'].values[i].lower()
        params['priors'].values[i] = params['priors'].values[i].strip('][').split(', ')
        params['population'].values[i] = int(params['population'].values[i])
        
        if params['distribution'].values[i] == 'fixed':
            params['priors'].values[i] = [float(params['priors'].values[i][0])]
            
        elif params['distribution'].values[i] == 'uniform':
            params['priors'].values[i] = [float(params['priors'].values[i][0]),float(params['priors'].values[i][1])]
        
    return params, number_of_samples

# Function to generate all the permutations for a given set of base parameters from a JSON file
@decorator_get_params
def load_from_json(path, number_of_samples='All'):

    params_data = json.loads(open(path, "r").read())
    params = pd.DataFrame(params_data['parameters'])

    return params, number_of_samples
   





