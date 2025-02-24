import numpy as np
from functions_for_pytest import *

mesh, x, d, f, r = run_test_x_axial()

print(np.linalg.norm(d - x) <= 1e-6)
print(np.linalg.norm(r - f) <= 1e-6)

mesh, x, d, f, r = run_test_xyz_axial()

print(np.linalg.norm(d - x) <= 1e-6)
print(np.linalg.norm(r - f) <= 1e-6)