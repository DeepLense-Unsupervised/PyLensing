import numpy as np
import autolens as al
import matplotlib.pyplot as plt
import math
import scipy.io
import h5py
import os
from progress.bar import Bar
import itertools
import random

def gen_data(parameters,
             pixel_scales=0.1,
             psf_shape=[11,11],
             psf_sigma=0.1,
             grid_sub_size=2,
             grid_shape=[100,100],
             output_type='image',
             output_path='./lens_sub_spherical',
             file_name='particle'):
             
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    bar = Bar('Processing lensing images', max=parameters.shape[0])
    lensing_images = []
    
    
    for i in range(parameters.shape[0]):
    
            params = parameters[i]
            psf = al.Kernel.from_gaussian(shape_2d=(psf_shape[0], psf_shape[1]), sigma=psf_sigma, pixel_scales=pixel_scales)
            grid = al.Grid.uniform(shape_2d=(grid_shape[0], grid_shape[1]), pixel_scales=pixel_scales, sub_size=grid_sub_size)

            vortex_profiles = []
            
            # Dark Matter Halo
            vortex_profiles.append(("dmh_profile",al.mp.SphericalIsothermal(centre=(params[8], params[9]), einstein_radius=params[10])))
                   
            # Get positional parameters for particle substructure
            rad_dist = np.random.uniform( params[25], params[26], int(params[23]) ).tolist()
            ang_pos = np.random.uniform( params[27], params[28], int(params[24]) ).tolist()
            
            pos_args = list(list(itertools.product(rad_dist, ang_pos)))
            random.shuffle(pos_args)
            
            # Linear mass distribution for substructure (string of mass on galactic scales)
            for j in range(int(params[21])):
                
                x0 = params[0] + pos_args[j][0]*math.cos(pos_args[j][1])
                y0 = params[1] + pos_args[j][0]*math.sin(pos_args[j][1])
                
                vortex_profiles.append(("point_mass_profile_" + str(j+1),
                al.mp.PointMass(centre=(x0,y0), einstein_radius=(params[22]/params[21])**0.5 * params[10])
                ))
        
            # Lens galaxy
            lensing_galaxy = al.Galaxy(
                redshift=params[2],
                # Light Profile
                light=al.lp.EllipticalSersic(
                    centre=(params[0], params[1]),
                    axis_ratio=params[3],
                    phi=params[4],
                    intensity=params[5],
                    effective_radius=params[7],
                    sersic_index=params[6],
                ),
                # Mass Profile
                **dict(vortex_profiles),
                
                # External Shear
                shear=al.mp.ExternalShear(magnitude=params[11], phi=params[12]),
            )

            galaxies=[lensing_galaxy]
            
            # Calculate coordinates of lensed galaxy
            x = params[0] + params[13]*math.cos(params[14])
            y = params[1] + params[13]*math.sin(params[14])
                        
            # Source galaxy
            lensed_galaxy = al.Galaxy(
                redshift=params[15],
                # Light Profile
                light=al.lp.EllipticalSersic(
                    centre=(x, y),
                    axis_ratio=params[16],
                    phi=params[17],
                    intensity=params[18],
                    effective_radius=params[20],
                    sersic_index=params[19],
                ),
            )

            galaxies.append(lensed_galaxy)

            tracer = al.Tracer.from_galaxies(galaxies)
            
            simulator = al.SimulatorImaging(
                exposure_time_map=al.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
                psf=psf,
                background_sky_map=al.Array.full(fill_value=0.1, shape_2d=grid.shape_2d),
                add_noise=True,
            )

            imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)
            image = imaging.image.in_2d

            if output_type.lower() == 'image':
            
                output_file = os.path.join(output_path, file_name + str(i+1) + '.png')
                plt.imsave(output_file, image, cmap='gray')
                
            if output_type.lower() in ( 'numpy' , 'matlab' , 'hdf5' ) :
                lensing_images.append(image)
                
            bar.next()
                    
    bar.finish()
    
    lensing_images = np.asarray(lensing_images)
    
    if output_type.lower() == 'numpy':
        
        output_file = os.path.join(output_path, file_name + '.npy')
        np.save(output_file, lensing_images)
        print('Dimensions of the data: {}'.format(lensing_images.shape))
    
    if output_type.lower() == 'matlab':
        
        output_file = os.path.join(output_path, file_name + '.mat')
        scipy.io.savemat(output_file, mdict={'vortex': lensing_images})
        print('Dimensions of the data: {}'.format(lensing_images.shape))
        
    if output_type.lower() == 'hdf5':
    
        output_file = os.path.join(output_path, file_name + '.h5')
        with h5py.File(output_file, 'w') as hf:
            hf.create_dataset("vortex",  data=lensing_images)
        print('Dimensions of the data: {}'.format(lensing_images.shape))





