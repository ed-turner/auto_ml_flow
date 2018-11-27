from setuptools import setup

__version__ = '0.1'

with open("README.txt") as f:
    README = f.readlines()

with open("LICENSE") as f:
    LICENSE = f.readlines()

REQUIREMENTS = ['pandas', 'numpy']

setup(
    name='auto_ml_flow',
    packages=['auto_ml_flow'],
    version=__version__,
    install_requires=REQUIREMENTS,
    extras_require={
        'complete':  ["xgboost", "lgbm"],
        'xgb': ["xgboost"],
        'lgbm': ['lgbm']
    },
    author="Edward Turner",
    author_email="edward.turnerr@gmail.com",
    description=README,
    license=LICENSE
)
