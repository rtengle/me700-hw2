# If you cannot run the jupyter notebook, I highly recommend figuring out how to get that to run.
# This is just a more basic version of the content inside that notebook.

# There are two main class associated with building the truss system

# Mesh: This class represents the overall structure of the system and is defined as a set of nodes
# connected by elements. The nodes act as the boundary conditions of the system and are where external
# forces and displacements are made.
from mesh import Mesh

# Beam: This class represents the individual beams comprising the system. The domain variables such as
# area, elastic modulus, etc. are defined on this element (apart from length) while the boundary and
# loading conditions are defined on the nodes.
from beam import Beam

# We also require numpy datatypes to run
import numpy as np

# Let's start with a slightly nuanced system: A simply-supported beam. This has two fixtures that allow for
# rotation while the load is applied at the center. To simulate this, we will discretize it to two domains
# connected where the force is being applied. The rotation at the center will be set to zero, but it will be
# free to move around translationally.

# We start with initializing our mesh with a specified nodal DoF. For beams, there are 6: 3 translational and
# 3 rotational. This is just because the Mesh class can be used for things other than beams.

mesh = Mesh(6)

# We will now add our nodes. All of them will sit along the x-axis and will have the same properties. We start
# by defining our system parameters:

F = 1
E = 1
A = 1
L = 1
Iy = 1
Iz = 1
J = 1
v = 0.2

# Nodes are added to the mesh with their position, boundary conditions, and applied forces/moments each passed as an
# array. Boundary conditions follow the format where the a number represents a fixed displacement while np.nan
# means that the node is free to move along that axis. 

# For our system, the beam is fixed translationally at the ends and fixed rotationally at the center. Nodes are identified
# with an integer from 0 to N. No gaps should be present otherwise there are issues.

# If no applied forces are given, it is assumed to be zero. Displacement format follows [x, y, z, rotx, roty, rotz]
mesh.add_node(0, np.array([0, 0, 0]), np.array([0, 0, 0, np.nan, np.nan, np.nan]))
# Center node. Applied force format follow [Fx, Fy, Fz, Mx, My, Mz]
mesh.add_node(1, np.array([L/2, 0, 0]), np.array([np.nan, np.nan, np.nan, 0, 0, 0]), bf=np.array([0, -F, 0, 0, 0, 0]))
# End node. Same as the first node just at a different position.
mesh.add_node(2, np.array([L, 0, 0]), np.array([0, 0, 0, np.nan, np.nan, np.nan]))

# We now add our beams. We can define a simple beam object and reuse it for both connections.
# The beam is initialized using all the required variables along with an orientation vector. This is just a vector
# that lies in the xy plane and is used to orient the beam in case it's asymmetrical along the xy and/or xz planes.

beam = Beam(E=E, A=A, Iy=Iy, Iz=Iz, J=J, v=v, y=np.array([0, 1, 0]))
# Add the connection to the mesh as a beam. We specify the nodes based on the identifier
mesh.add_element(beam, 0, 1)
mesh.add_element(beam, 1, 2)

# We have now defined our system, so all that's left is to solve it:

# Outputs the global displacement and force vectors respectively. Post-processing can separate it out.
x, f = mesh.solve()

# In this case, we know what our displacements d and our reactions r should be, so we can compare them:

tzmax = F*L**2/(16*E*Iz)
ymax = -F*L**3/(48*E*Iz)

d = np.array([
    0, 0, 0,
    0, 0, -tzmax,
    0, ymax, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, tzmax
])
r = np.array([
    0, F/2, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, F/2, 0,
    0, 0, 0
])

print(np.linalg.norm(d - x))
print(np.linalg.norm(r - f))

# Have fun. Note that this code currently has like zero error handling because it took so long to set up so be careful.