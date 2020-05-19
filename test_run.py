# Example Code to test docker image

from lensing import gen_params_vortex as genpv
from lensing import gen_params_base as genpb
from lensing import gen_params_spherical as genps
from lensing import get_params as getp
from lensing import gen_data_vortex as gendv
from lensing import gen_data_base as gendb
from lensing import gen_data_spherical as gends
import pandas as pd

# Generate Lensing Images with No Substructure

print('Generating Lensing Images with No Substructure')
generator = genpb.gen_params(save_params=False)
base_parameters = generator.run()
params = getp.get_params(base_parameters, number_of_samples=5)
print('Dimensions of parameter space: {}'.format(params.shape))
gendb.gen_data(params,output_type='Image', grid_shape=[150,150], pixel_scales=0.05)
print('Done!')

'''
# Generate Lensing Images with Vortex Substructure

print('Generating Lensing Images with Vortex Substructure')
generator = genpv.gen_params(save_params=False)
base_parameters = generator.run()
params = getp.get_params(base_parameters, number_of_samples=5)
print('Dimensions of parameter space: {}'.format(params.shape))
gendv.gen_data(params,output_type='Image', grid_shape=[150,150], pixel_scales=0.05)
print('Done!')

# Generate Lensing Images with Spherical (Particle) Substructure

print('Generating Lensing Images with Spherical (Particle) Substructure')
generator = genps.gen_params(save_params=False)
base_parameters = generator.run()
params = getp.get_params(base_parameters, number_of_samples=5)
print('Dimensions of parameter space: {}'.format(params.shape))
gends.gen_data(params,output_type='Image', grid_shape=[150,150], pixel_scales=0.05)
print('Done!')
'''





