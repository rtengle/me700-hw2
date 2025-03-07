[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)

[![codecov](https://codecov.io/gh/rtengle/me700-hw2/graph/badge.svg?token=V8BG4FHMD7)](https://codecov.io/gh/rtengle/me700-hw2)
[![tests](https://github.com/rtengle/me700-hw2/actions/workflows/tests.yml/badge.svg)](https://github.com/rtengle/me700-hw2/actions)

# me700-hw2
Basic 3D mesh-based matrix element mechanics solver. Includes additional class for solving beam problems. 

## Nodes, Elements, and Meshes

The most basic form of an element is as follows:

```math
n_0 \xrightarrow{E_{0}} n_1
```

Here we have an element $E_0$ connected by two nodes: $n_0$ and $n_1$. Each node has a set of degrees of freedom represented by $\vec{x}$ and a set of actions or forces on each degree of freedom represented by $\vec{f}$. The element relates the two nodes using the equation:

$$ \left[\begin{matrix} \vec{f}_0 \\ \vec{f}_1 \end{matrix}\right] + \vec{f}_b = K_0 \left[\begin{matrix} \vec{x}_0 \\ \vec{x}_1 \end{matrix}\right]$$

where $K_0$ is the stiffness matrix of element, $\vec{f}_b$ is any body force contributions, $\vec{f}_n$ is the internal reaction at the node, and $\vec{x}_n$ is the displacement of the node in each degree of freedom.

A mesh describes a connection of elements and nodes. An example is as follows:

$$n_0 \xrightarrow{E_{0}} n_1 \xrightarrow{E_{1}} n_2$$

Here we have two elements sharing a node. A mesh is used to describe this connection of elements and nodes. The only condition is that the nodes must have the same degrees of freedom. Stiffness matrices and boudnary conditions can differ for each element and ndoe.

Once this mesh is defined, the entire system is solved by linearly assembling all the contributions into a single linear algebra problem:

$$\left[\begin{matrix} \vec{f}_0 \\ \vec{f}_1 \\ \vdots \\ \vec{f}_N \end{matrix}\right] + \vec{f}_b = K \left[\begin{matrix} \vec{x}_0 \\ \vec{x}_1 \\ \vdots \\ \vec{x}_N \end{matrix}\right]$$

where K is assembled based on the element connections in the mesh. Once this is set along with the boundary conditions, the system is solved for the total force and displacement vectors.

These are defined using the ```Element``` and ```Mesh``` classes.

## Euler-Bernoulli Beams

In addition to the basic ```Element``` and ```Mesh``` classes, a subclass of ```Element``` is included known as ```Beam```. This defines a Euler-Bernoulli beam element in a structure. This element has the following degrees of freedom and forces:

$$\left[\begin{matrix} F_x \\ F_y \\ F_z \\ M_x \\ M_y \\ M_z \end{matrix}\right] = K \left[\begin{matrix} u \\ v \\ w \\ \theta_x \\ \theta_y \\ \theta_z \end{matrix}\right]$$

The boundary conditions for this are the forces & moments applied at each node and the set displacement and/or rotation of the node. ```Mesh``` can be used with ```Beam``` as an element as long as the user defines the node degrees of freedom as 6. See tutorial for how to use.

## Global Eigenvalue Study

Finally, the ```Mesh``` class supports the ability to perform a global eigenvalue study using the ```global_eigenvalue_study``` method. This takes the form:

$$A\vec{\delta} = \lambda B \vec{\delta}$$

where $A$ and $B$ are matrices assembled from the elements and $\lambda$ and $\delta$ represent the eigenvalue and eigenvectors respectively for the given system boundary conditions. Note that this study currently does not support non-zero displacement boundary conditions. All that needs to be passed into the study is a function that calculates the $A$ and $B$ contributions from each element.

The ```Beam``` class includes the necessary function for a buckling study. A buckling study determines how much additional load must be applied to the structure before it starts buckling based on the following eigenvalue problem:

$$K \vec{x}_b = -\lambda K_g \vec{x}_b$$

where $K_g$ is the effects of geometric displacements of the forces, $\lambda$ is the multiplicative factor for how much additional load before a specific buckling mode is activated, and $\vec{x}_b$ represents the deformation resulting from the corresponding mode.

A buckling study is performed by passing ```Beam.beam_buckling_eigenmatrix``` into the  ```global_eigenvalue_study``` method. See tutorial for more details.

# Installation

To install this package, please begin by setting up a conda environment (mamba also works):
```bash
conda create --name me700-hw2-env python=3.12
```
Once the environment has been created, activate it:

```bash
conda activate me700-hw2-env
```
Double check that python is version 3.12 in the environment. It should still work on a later version, but it was made on this one.
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
Code coverage should be above 90%.

