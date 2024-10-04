import os
from setuptools import setup, find_packages


# Path to the current directory
here = os.path.abspath(os.path.dirname(__file__))

# Read the required dependencies from the requirements.txt
with open('requirements.txt') as f:
    required_dependencies = f.read().splitlines()


# Setup function
setup(
    name='TP1_IMN708',
    version='0.1.0',
    description='A short description of your project',
    author='Chan Nam Nguyen, Zineb El yamani',
    url='https://github.com/elyz081/TP1_IMN708.git',
    packages=find_packages(),
    install_requires=required_dependencies,
    entry_points={
        'console_scripts': [
            'view_image = scripts.view_image:main',
            'mip_MIP = scripts.compute_mip_MIP:main',
            'stats = scripts.show_image_stats:main',
            'denoise = scripts.denoise_image:main',
        ],
    },
)
