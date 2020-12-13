from setuptools import find_packages
from setuptools import setup

setup(
    name='data-generator-for-treatment-effect',
    version='1.0.0',
    description='Simulation data generation from [Powers, 2018]',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)