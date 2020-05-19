# Author: Pranath Reddy
# This module is for generating a default set of parameters for galaxy-galaxy strong lensing with vortex substructure

import numpy as np
import pandas as pd
import json

class gen_params:

    params_values = '''
    
            {   "parameters": [

            {"profile": "Lensing Galaxy - Sersic Light Profile",
            "parameter": "x_pos",
            "distribution": "fixed",
            "population": 1,
            "priors": [0]},
                 
            {"profile": "Lensing Galaxy - Sersic Light Profile",
            "parameter": "y_pos",
            "distribution": "fixed",
            "population": 1,
            "priors": [0]},
            
            {"profile": "Lensing Galaxy - Sersic Light Profile",
            "parameter": "redshift",
            "distribution": "fixed",
            "population": 1,
            "priors": [0.5]},
            
            {"profile": "Lensing Galaxy - Sersic Light Profile",
            "parameter": "axis_ratio",
            "distribution": "uniform",
            "population": 5,
            "priors": [0.5,1.0]},
            
            {"profile": "Lensing Galaxy - Sersic Light Profile",
            "parameter": "orientation",
            "distribution": "uniform",
            "population": 5,
            "priors": [0,6.28318530718]},
            
            {"profile": "Lensing Galaxy - Sersic Light Profile",
            "parameter": "intensity",
            "distribution": "fixed",
            "population": 1,
            "priors": [1.2]},
            
            {"profile": "Lensing Galaxy - Sersic Light Profile",
            "parameter": "sersic_index",
            "distribution": "fixed",
            "population": 1,
            "priors": [2.5]},
            
            {"profile": "Lensing Galaxy - Sersic Light Profile",
            "parameter": "eff_radius",
            "distribution": "fixed",
            "population": 1,
            "priors": [0.5]},
            
            {"profile": "Dark Matter Halo – Spherical Isothermal",
            "parameter": "x_pos",
            "distribution": "fixed",
            "population": 1,
            "priors": [0]},
                 
            {"profile": "Dark Matter Halo – Spherical Isothermal",
            "parameter": "y_pos",
            "distribution": "fixed",
            "population": 1,
            "priors": [0]},
            
            {"profile": "Dark Matter Halo – Spherical Isothermal",
            "parameter": "einstein_radius",
            "distribution": "fixed",
            "population": 1,
            "priors": [1.2]},
            
            {"profile": "External Shear",
            "parameter": "magnitude",
            "distribution": "uniform",
            "population": 5,
            "priors": [0.0,0.3]},
            
            {"profile": "External Shear",
            "parameter": "angle",
            "distribution": "uniform",
            "population": 5,
            "priors": [0,6.28318530718]},
            
            {"profile": "Lensed Galaxy - Sersic Profile",
            "parameter": "radial_distance",
            "distribution": "uniform",
            "population": 5,
            "priors": [0.0,1.2]},
                 
            {"profile": "Lensed Galaxy - Sersic Profile",
            "parameter": "angular_position",
            "distribution": "uniform",
            "population": 5,
            "priors": [0,6.28318530718]},
            
            {"profile": "Lensed Galaxy - Sersic Profile",
            "parameter": "redshift",
            "distribution": "fixed",
            "population": 1,
            "priors": [1.0]},
            
            {"profile": "Lensed Galaxy - Sersic Profile",
            "parameter": "axis_ratio",
            "distribution": "uniform",
            "population": 5,
            "priors": [0.7,1.0]},
            
            {"profile": "Lensed Galaxy - Sersic Profile",
            "parameter": "orientation",
            "distribution": "uniform",
            "population": 5,
            "priors": [0,6.28318530718]},
            
            {"profile": "Lensed Galaxy - Sersic Profile",
            "parameter": "intensity",
            "distribution": "uniform",
            "population": 5,
            "priors": [0.7,0.9]},
            
            {"profile": "Lensed Galaxy - Sersic Profile",
            "parameter": "sersic_index",
            "distribution": "fixed",
            "population": 1,
            "priors": [1.5]},
            
            {"profile": "Lensed Galaxy - Sersic Profile",
            "parameter": "eff_radius",
            "distribution": "fixed",
            "population": 1,
            "priors": [0.5]},
            
            {"profile": "Vortex",
            "parameter": "x_pos",
            "distribution": "fixed",
            "population": 1,
            "priors": [0]},
                 
            {"profile": "Vortex",
            "parameter": "y_pos",
            "distribution": "fixed",
            "population": 1,
            "priors": [0]},
            
            {"profile": "Vortex",
            "parameter": "length",
            "distribution": "fixed",
            "population": 1,
            "priors": [1]},
            
            {"profile": "Vortex",
            "parameter": "orientation",
            "distribution": "uniform",
            "population": 5,
            "priors": [0,6.28318530718]},
            
            {"profile": "Vortex",
            "parameter": "total_mass",
            "distribution": "fixed",
            "population": 1,
            "priors": [0.01]}
    ]}'''
    
    params_data = json.loads(params_values)
    
    params = pd.DataFrame(params_data['parameters'])
    
    # Constructor
    def __init__(self, save_params, save_type='json', output_path='./'):
        self.output_path = output_path
        self.save_params = save_params
        self.save_type = save_type
        
        '''
        
        Args:
        ______
        
        save_params: bool
           Will save the parameters to specified save type if set to True
           
        save_type: str ('csv', 'json')
        
        output_path: str
        '''
        
    def run(self):
        
        if self.save_params == True:
            
            if self.save_type.lower() == 'csv':
                gen_params.params.to_csv(self.output_path + '/params_vortex.csv', index = False, header=True)
                
            if self.save_type.lower() == 'json':
                out_file = open(self.output_path + '/params_vortex.json', "w")
                json.dump(gen_params.params_data, out_file, indent = 2)
            
        return gen_params.params
    





