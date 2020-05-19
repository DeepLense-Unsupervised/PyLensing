# Author: Pranath Reddy
# This module is for generating galaxy-galaxy strong lensing images with no substructure

import numpy as np
import autolens as al
import matplotlib.pyplot as plt
import math
import scipy.io
import h5py
import os
from progress.bar import Bar

def gen_data(parameters,
             pixel_scales=0.1,
             psf_shape=[11,11],
             psf_sigma=0.1,
             grid_sub_size=2,
             grid_shape=[100,100],
             output_type='image',
             output_path='./lens_nosub',
             file_name='base'):
             
    '''
     
     Args:
     ______
     
     pixel_scales: float
        The arc-second to pixel conversion factor of each pixel.
     
     psf_shape: []
        Shape of the Gaussian kernel
     
     psf_sigma: float
        Standard deviation for Gaussian kernel
     
     grid_sub_size: int
        The size (sub_size x sub_size) of each unmasked pixels sub-grid.
     
     grid_shape: []
     
     output_type: str
        'image': save the lensing images as .png files
        'numpy': save the lesning images as a numpy array
        'matlab': save the lesning images as a matlab (.MAT) file
        'hdf5': save the lensing images as a HDF file
     
     output_path: str
     
     file_name: str
     
    '''
             
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    bar = Bar('Processing lensing images', max=parameters.shape[0])
    lensing_images = []
    for i in range(parameters.shape[0]):
    
            params = parameters[i]
            psf = al.Kernel.from_gaussian(shape_2d=(psf_shape[0], psf_shape[1]), sigma=psf_sigma, pixel_scales=pixel_scales)
            grid = al.Grid.uniform(shape_2d=(grid_shape[0], grid_shape[1]), pixel_scales=pixel_scales, sub_size=grid_sub_size)

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
                # Dark Matter Halo
                mass=al.mp.SphericalIsothermal(centre=(params[8], params[9]), einstein_radius=params[10]
                ),
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
            
            # Export all the Lensing Images
            if output_type.lower() == 'image':
            
                output_file = os.path.join(output_path, file_name + str(i+1) + '.png')
                plt.imsave(output_file, image, cmap='gray')
                
            if output_type.lower() in ( 'numpy' , 'matlab' , 'hdf5' ) :
                lensing_images.append(image)
                
            bar.next()
                    
    bar.finish()
    
    lensing_images = np.asarray(lensing_images)
    
    # Dump all the Lensing Images into a numpy array
    if output_type.lower() == 'numpy':
    
        output_file = os.path.join(output_path, file_name + '.npy')
        np.save(output_file, lensing_images)
        print('Dimensions of the data: {}'.format(lensing_images.shape))
    
    # Dump all the Lensing Images into a matlab (.MAT) file
    if output_type.lower() == 'matlab':
        
        output_file = os.path.join(output_path, file_name + '.mat')
        scipy.io.savemat(output_file, mdict={'vortex': lensing_images})
        print('Dimensions of the data: {}'.format(lensing_images.shape))
        
    # Dump all the Lensing Images into a HDF file
    if output_type.lower() == 'hdf5':
    
        output_file = os.path.join(output_path, file_name + '.h5')
        with h5py.File(output_file, 'w') as hf:
            hf.create_dataset("vortex",  data=lensing_images)
        print('Dimensions of the data: {}'.format(lensing_images.shape))





