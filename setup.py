import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open('requirements.txt') as f:
    required_dependencies = f.read().splitlines()
    external_dependencies = []
    torch_added = False
    for dependency in required_dependencies:
        external_dependencies.append(dependency)

setup(
    name='TP1_IMN708',
    version='0.1.0',
    description='A short description of your project',
    author='Chan Nam Nguyen, Zineb El yamani',
    url='https://github.com/elyz081/TP1_IMN708.git',
    packages=find_packages(),
    install_requires=external_dependencies,
    entry_points={
        'console_scripts': [
            'view_image = scripts.view_image:main',
            'compute_mip_MIP = scripts.compute_mip_MIP:main',
            'stats = scripts.show_image_stats:main',
            'denoise = scripts.denoise_image:main',
            'show_histogram = scripts.show_histogram:main',
        ],
    },
)
