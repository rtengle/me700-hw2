# me700-hw2
Basic 3D mesh-based matrix element mechanics solver. Includes additional class for solving beam problems. See tutorial for how to run.

# Installation

To install this package, please begin by setting up a conda environment (mamba also works):
```bash
conda create --name me700-hw1-env python=3.12
```
Once the environment has been created, activate it:

```bash
conda activate me700-hw1-env
```
Double check that python is version 3.9.18 in the environment. It should still work on a later version, but it was made on this one.
```bash
python --version
```
Ensure that pip is using the most up to date version of setuptools:
```bash
pip install --upgrade pip setuptools wheel
```
Create an editable install of the bisection method code (note: you must be in the correct directory):
```bash
pip install -e .
```
Test that the code is working with pytest:
```bash
pytest -v --cov=. --cov-report=xml
```
Code coverage should be 100%. Now you are prepared to write your own code based on this method and/or run the tutorial.

