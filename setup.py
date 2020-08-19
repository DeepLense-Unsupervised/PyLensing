from distutils.core import setup
setup(
  name = 'PyLensing',
  packages = ['lensing'],
  version = '0.1',
  license='MIT',
  description = 'A tool for generating lensing images based on PyAutoLens simulations',
  author = 'K Pranath Reddy',
  author_email = 'pranath.mail@gmail.com',
  url = 'https://github.com/DeepLense-Unsupervised/PyLensing',
  keywords = ['Gravitational Lensing', 'Simulation', 'Dark Matter'],
  install_requires=[            
          'numpy==1.18.3',
          'scipy==1.4.1',
          'joblib==0.14.1',
          'Cython==0.29.16',
          'pandas==1.0.3',
          'progress==1.5',
          'pymultinest==2.9',
          'autolens==0.46.2',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
  ],
)
