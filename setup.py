from setuptools import setup, find_packages

setup(
    name='BoostMut',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'BoostMut': ['benchmarks/*.csv'],
    },
    install_requires=[
        'numpy',
        'pandas',
        'MDAnalysis',
        'pydssp',
        'freesasa',],
    entry_points={
        'console_scripts':['boostmut=BoostMut.run_BoostMut:main']
    },
    author='Kerlen T. Korbeld',
    description='A package for analyzing the stabilizing effect of mutations in short MD simulations',
)
