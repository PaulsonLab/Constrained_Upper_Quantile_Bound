"""Install Constrained Upper Quantile Bound (CUQB)"""

from setuptools import find_packages
from setuptools import setup

setup(
    name='Constrained Upper Confidence Bound',
    version='1.0',
    description=(
        'Constrained Upper Quantile Bound (CUQB) for solving constrained hybrid models.'),
    author='Congwen Lu and Joel Paulson',
    author_email='Paulson.82@osu.edu',
    url='https://github.com/PaulsonLab/Constrained_Upper_Quantile_Bound',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'numba',
        'botorch',
        'cyipopt',
	'fast_soft_sort'
    ]
)