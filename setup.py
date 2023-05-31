from setuptools import setup, find_packages

setup(
    name='2D Shapley',
    version='2.0',
    description='Packages for 2D Shapley: A Framework for Fragmented Data Valuation',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'copy',
        'numpy',
        'scipy',
        'matplotlib',
        'multiprocessing',
        'scikit-learn',
        'pandas',
        'pickle'
    ],
    include_package_data=True
)