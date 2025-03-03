import math_utils as mu
import numpy as np
import scipy.linalg as spla

E = 1000
r = 1
L = 20
v = 0.3
Iy = np.pi * r**4 / 4
A = np.pi * r**2
Iz = Iy
J = Iy*2

print(np.pi**2 * E * Iy/(4*L**2))

K = mu.local_elastic_stiffness_matrix_3D_beam(E=E, nu=v, A=A, L=L, Iy=Iy, Iz=Iz, J=J)
Kg = mu.local_geometric_stiffness_matrix_3D_beam(L=L, A=A, I_rho=J, Fx2=-1, My1=0, My2=0, Mz1=0, Mz2=0, Mx2=0)
a, b = spla.eig(a=K, b=-Kg)
print(np.sort(a))
