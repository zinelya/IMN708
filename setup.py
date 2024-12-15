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
            'view_image = scripts.TP1.view_image:main',
            'compute_mip_MIP = scripts.TP1.compute_mip_MIP:main',
            'stats = scripts.TP1.show_image_stats:main',
            'denoise = scripts.TP1.denoise_image:main',
            'compare_pairs = scripts.TP2.compare_pairs:main',
            'transform_grid = scripts.TP2.transform_grid:main',
            'register = scripts.TP2.register_image:main',
            'compute_dti_metrics = scripts.TP3.compute_dti_metrics:main',
            'track_dti = scripts.TP3.track_dti:main',
            'track_fodf = scripts.TP3.track_fodf:main',
            'show_q_space = scripts.TP3.show_q_space:main',
        ],
    },
)