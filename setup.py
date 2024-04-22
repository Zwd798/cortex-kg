from setuptools import setup, find_packages

setup(
    name='cortex-kg',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ],
    entry_points={
        'console_scripts': [
            'cortex-kg = cortex-kg.cli:main',
        ],
    },
)