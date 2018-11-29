from setuptools import setup

__version__ = '0.1'

with open("README.txt") as f:
    README = f.readlines()

with open("LICENSE") as f:
    LICENSE = f.readlines()

REQUIREMENTS = ['pandas==0.23.4', 'numpy==1.15.4', 'sklearn==0.0', 'scipy==1.1.0']

setup(
    name='auto_ml_flow',
    packages=['auto_ml_flow', ],
    version=__version__,
    install_requires=REQUIREMENTS,
    author="Edward Turner",
    author_email="edward.turnerr@gmail.com",
    description=README,
    license=LICENSE
)
