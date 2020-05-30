# Example Code to test docker image

from lensing import gen_params_vortex as genpv
from lensing import gen_params_base as genpb
from lensing import gen_params_spherical as genps
from lensing import get_params as getp
from lensing import gen_data_vortex as gendv
from lensing import gen_data_base as gendb
from lensing import gen_data_spherical as gends
import pandas as pd
import numpy as np

'''
mass_fractions = np.random.uniform(low=0.001, high=0.003, size=(25,))

# Generate Lensing Images with Vortex Substructure

print('Generating Lensing Images with Vortex Substructure')
generator = genpv.gen_params(save_params=False)
base_parameters = generator.run()
params = getp.get_params(base_parameters, number_of_samples=5)
print('Dimensions of parameter space: {}'.format(params.shape))
gendv.gen_data(params,output_type='Image', grid_shape=[150,150], sub_halo_mass_fractions=mass_fractions, pixel_scales=0.05)
print('Done!')

# Generate Lensing Images with Spherical (Particle) Substructure

print('Generating Lensing Images with Spherical (Particle) Substructure')
generator = genps.gen_params(save_params=False)
base_parameters = generator.run()
params = getp.get_params(base_parameters, number_of_samples=5)
print('Dimensions of parameter space: {}'.format(params.shape))
gends.gen_data(params,output_type='Image', grid_shape=[150,150], sub_halo_mass_fractions=mass_fractions, pixel_scales=0.05)
print('Done!')
'''

#masses = np.random.uniform(low=0.8e10, high=1.2e10, size=(25,))
masses = getp.draw_sub_masses(25,0.8e10,1.2e10)

# Generate Lensing Images with Vortex Substructure

print('Generating Lensing Images with Vortex Substructure')
generator = genpv.gen_params(save_params=False)
base_parameters = generator.run()
params = getp.get_params(base_parameters, number_of_samples=5)
print('Dimensions of parameter space: {}'.format(params.shape))
gendv.gen_data(params,output_type='Image', grid_shape=[150,150], sub_halo_mass=masses, pixel_scales=0.05)
print('Done!')

# Generate Lensing Images with Spherical (Particle) Substructure

print('Generating Lensing Images with Spherical (Particle) Substructure')
generator = genps.gen_params(save_params=False)
base_parameters = generator.run()
params = getp.get_params(base_parameters, number_of_samples=5)
print('Dimensions of parameter space: {}'.format(params.shape))
gends.gen_data(params,output_type='Image', grid_shape=[150,150], sub_halo_mass=masses, pixel_scales=0.05)
print('Done!')






