from beam import Beam
from mesh import Mesh, Node
from networkx import Graph
import numpy as np

mesh = Mesh(6)

mesh.add_node(0, np.array([0,0,0]), np.zeros(6))
mesh.add_node(1, np.array([1,0,0]), np.full(6, np.nan), bf=np.array([0, 1, 0, 0, 0, 0]))

beam = Beam(E=100, A=1, Iy = 1, Iz = 1, J = 1, v = 0.2, y=np.array([0, 1, 0]))
mesh.add_element(beam, 0, 1)

x, f = mesh.solve()

print(x)
print(f)