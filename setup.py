from setuptools import setup, find_packages

setup(
    name='emg2qwerty',
    version='0.1.0',
    description='A project for predicting QWERTY keystrokes from EMG signals.',
    author='CTRL Research Team, Meta Reality Labs',
    author_email='agramfort@meta.com',
    packages=find_packages(),
    install_requires=[
        # Left empty so you use the conda environment.yml file
    ],
)