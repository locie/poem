from setuptools import setup, find_packages

setup(
    name='poem',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        dash,
    ],
    entry_points={
        'console_scripts': [
            # Scripts d'entrée si nécessaire
        ],
    },
)
