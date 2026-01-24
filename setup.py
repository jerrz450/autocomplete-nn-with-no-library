from setuptools import setup, find_packages

setup(
    name='autopredict',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
    ],
    python_requires='>=3.7',
)
