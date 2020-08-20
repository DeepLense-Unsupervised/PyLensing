# PyLensing
A tool for generating lensing images based on PyAutoLens simulations

[Example Notebook](https://github.com/DeepLense-Unsupervised/PyLensing/blob/master/Example_Notebook.ipynb)

![Lensing Example](https://github.com/PyJedi/PyLensing/blob/master/gitimage.png)

### Docker Specific Instructions
___

1. Install Docker, clone this repository
```
git clone https://github.com/PyJedi/PyLensing.git
```
2. CD into the repository and build the docker image
```
cd PyLensing
docker build -t image .
```
3. Start the docker image
```
docker run -it image bash
```
4. clone the autolens workspace and set the WORKSPACE environment and PYTHONPATH
```
. ./setup_docker.sh
```
5. Run the test code
```
python test_run.py
```


