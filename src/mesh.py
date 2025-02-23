from networkx import Graph
from typing import Callable
import numpy as np

def unit(v):
    # Adding in a function that should be a default in numpy: getting the unit vector
    return v/np.linalg.norm(v)

class Node():
    def __init__(self, pos: np.ndarray, bc: np.ndarray, bf = None):
        # Initializes node using a position vector with a set of displacement boundary conditions. NaN is considered free.
        self.pos = pos
        self.bc = bc
        if type(bf) != np.ndarray and bf == None:
            self.bf = np.ndarray(len(bc))
        else:
            self.bf = bf

    def add_bf(self, bf):
        self.bf = self.bf + bf

    def set_bc(self, bc: np.ndarray):
        self.bc = bc

    def add_bc(self, bc: float, dof: type[int | slice]):
        self.bc[dof] = bc

class Element():
    """A class representing an element inside of a MSA mesh. The element contains information pertaining to the constitutive equation:

    fn + fe = Kx

    where fn represents the nodal forces, fe represents external forces due to either prescribed load, body forces, etc, x represents
    the displacements of the node DOFs, and K is the stiffness matrix representing the linearized relationship between the forces and
    displacements. These contributions can depend on a state variable and a set of constant kwargs.

    Attributes
    ----------
    stiffness_method : Callable
        Method to calculate the local stiffness matrix K based on a state variable.
    body_forces_method : Callable
        Method to calculate the nodal body force contributions based on a state variable.
    transformation_method : Callable
        Method to calculate the given transformation matrix that transitions from the local system to the global system.
    orientation : ndarray
        3D unit vector in the xy plane used to orient the element.
    rotation : ndarray
        3 x 3 rotation matrix representing the change in coordinates systems from the local to the global
    
    """
    def __init__(self, stiffness: Callable, body_forces: type[int | Callable], transform: Callable, orientation: np.ndarray, **kwargs):
        """Constructs an Element based on a set of given functions and transformations.
        
        Parameters
        ----------
        stiffness : Callable
            Function that returns the stiffness based on a dynamic "state" variable.
        body_forces : Callable | int
            Either an int that states the DoF of the body forces or a function that updates the body forces based on a "state" variable.
        transform : Callable
            Function that returns the global matrix for the given DOFs. Externally specified because some DoFs may be scalar fields or transform weirdly.
        orientation : ndarray
            A 3D vector representing a vector in the xy-plane of the element. Used to construct the local rotation matrix.
        **kwargs
            Extra arguments that are passed to the stiffness and body_forces functions. These represent system constants throughout the analysis.
        """
        # Initializes element of mesh with a stiffness method and body method that are able to be dynamically updated along with an orientation vector.
        self.stiffness_method = stiffness
        self.body_forces_method = body_forces
        self.transformation_method = transform
        self.orientation = unit(orientation)

    def reorient(self, xvec, output=True):
        # Properly orients element based on a given length vector
        xhat = unit(xvec)
        zhat = unit(np.cross(xvec, self.orientation))
        yhat = np.cross(zhat, xhat)
        self.rotation = np.array([xhat, yhat, zhat]).T
        if output == True:
            return self.rotation

    def local_stiffness(self, nodes, state, output=True):
        # Updates the local stiffness matrix based on some state. Also outputs the results if desired.
        if state == None:
            self.local_stiffness_matrix = self.stiffness_method(nodes)
        else:
            self.local_stiffness_matrix = self.stiffness_method(nodes, state)
        if output == True:
            return self.local_stiffness_matrix
    
    def local_body_forces(self, nodes, state, output=True):
        # Updates the body force tuple based on some state. Also outputs the results if desired.
        if type(self.body_forces_method) == int:
            self.body_forces_pair = (np.zeros(self.body_forces_method), np.zeros(self.body_forces_method))
            if output == True:
                return self.body_forces_pair
        else:
            if state == None:
                self.body_forces_pair = self.body_forces_method(nodes)
            else:
                self.body_forces_pair = self.body_forces_method(nodes, state)
            if output == True:
                return self.body_forces_pair
    
    def global_transform(self, nodes, output=True):
        node1, node2 = nodes
        L = node2.pos - node1.pos
        self.global_transformation_matrix = self.transformation_method(self.reorient(L))
        if output == True:
            return self.global_transformation_matrix

        
    def global_coords(self, nodes: tuple, state=None):
        # Gets the global coordinate values for the stiffness and body forces
        K = self.local_stiffness(nodes, state)
        fb1, fb2 = self.local_body_forces(nodes, state)
        fb = np.append(fb1, fb2)
        G = self.global_transform(nodes)
        self.global_stiffness_matrix = G @ K @ G.T
        global_fb = (G @ fb)
        fg1, fg2 = np.split(global_fb, 2)
        self.global_body_forces = (fg1, fg2)
        return self.global_stiffness_matrix, self.global_body_forces



class Mesh(Graph):
    def __init__(self, dof):
        self.node_dof = dof
        super().__init__()

    def add_node(self, id, pos, bc=None, bf=None):
        if type(bc) != np.ndarray and bc == None:
            bc = self.dof
        super().add_node(node_for_adding=id, object=Node(pos=pos, bc=bc, bf=bf))

    def add_element(self, element: Element, n1, n2):
        super().add_edge(n1, n2, object=element)

    def add_stiffness(self, K_local, n1, n2):
        # Have to do it this way bc python is lame and doesn't allow you to append slices together.
        # Trying to do it with arrays also messes up the axes.
        n1_slice = slice(self.node_dof*n1, self.node_dof*(n1+1))
        n2_slice = slice(self.node_dof*n2, self.node_dof*(n2+1))
        self.K_total[n1_slice, n1_slice] += K_local[0:self.node_dof, 0:self.node_dof]
        self.K_total[n1_slice, n2_slice] += K_local[0:self.node_dof, self.node_dof:2*self.node_dof]
        self.K_total[n2_slice, n1_slice] += K_local[self.node_dof:2*self.node_dof, 0:self.node_dof]
        self.K_total[n2_slice, n2_slice] += K_local[self.node_dof:2*self.node_dof, self.node_dof:2*self.node_dof]

    def iterate_elements(self, states:type[dict | None] = None):
        """Iterates over the elements of the mesh, updates their key values, and adds their contributions to 
        the total stiffness matrix and connected nodes. 

        Attributes
        ----------
        states : dict | None
            Dictionary containing the non-constant state values of the elements. If None, then no state is passed.

        Returns
        -------
        K_total : ndarray
            Total ndarray for the full mesh.
        """
        for (n1, n2, el) in self.edges.data('object'):
            node_tuple = (self.nodes[n1]['object'], self.nodes[n2]['object'])
            if type(states) == dict:
                K_local, (fb1, fb2) = el.global_coords(node_tuple, states[el])
            else:
                K_local, (fb1, fb2) = el.global_coords(node_tuple, states)
            self.nodes[n1]['object'].add_bf(fb1)
            self.nodes[n2]['object'].add_bf(fb2)
            self.add_stiffness(K_local, n1, n2)

    def add_forces(self, i, node: Node):
        self.external_forces[i*self.node_dof:(i+1)*self.node_dof] += node.bf

    def set_boundary_conditions(self, i, node: Node):
        self.total_bc[self.node_dof*i:(i+1)*self.node_dof] = node.bc

    def iterate_nodes(self):
        for (i, node) in self.nodes.data('object'):
            self.add_forces(i, node)
            self.set_boundary_conditions(i, node)

    def get_shuffle_matrix(self):
        bc_sorted = np.argsort(self.total_bc)
        self.shuffle_matrix = np.zeros((self.total_dof, self.total_dof))
        self.bc_shuffle =  np.zeros(self.total_dof)
        for end_index, start_index in zip(range(self.total_dof), bc_sorted):
            self.shuffle_matrix[end_index, start_index] = 1
        self.bc_shuffle = self.total_bc[bc_sorted]

    def shuffle_system(self):
        self.get_shuffle_matrix()
        self.K_shuffle = self.shuffle_matrix.T @ self.K_total @ self.shuffle_matrix
        self.shuffled_forces = self.shuffle_matrix @ self.external_forces
        

    def assemble_global(self):
        # Iterate through edges, update them, grab their stiffness matrices, and sum them
        self.total_dof = self.node_dof*self.number_of_nodes()
        self.K_total = np.zeros((self.total_dof, self.total_dof))
        self.external_forces = np.zeros(self.total_dof)
        self.total_bc = np.full(self.total_dof, np.nan)
        self.iterate_elements()
        self.iterate_nodes()

    def solve_shuffle(self):
        # Find out when nan ends
        sep_index = np.where(np.isnan(self.bc_shuffle))[0][0] # approaches backwards

        # Break down system into two subequations
        forces_ef = self.shuffled_forces[sep_index:self.total_dof]
        xd = self.bc_shuffle[0:sep_index]
        Kfd = self.K_shuffle[sep_index:self.total_dof,0:sep_index]
        Kff = self.K_shuffle[sep_index:self.total_dof,sep_index:self.total_dof]

        # Solve for displacements
        xf = np.linalg.inv(Kff) @ (forces_ef - Kfd @ xd)
        # Solve for reactions
        
        self.x_shuffle = np.append(xd, xf)
        self.f_shuffle = self.K_shuffle @ self.x_shuffle - self.shuffled_forces

    def unshuffle_solution(self):
        self.x_total = self.shuffle_matrix.T @ self.x_shuffle
        self.f_total = self.shuffle_matrix.T @ self.f_shuffle

    def solve(self):
        # Assembles global matrix
        self.assemble_global()
        self.shuffle_system()
        self.solve_shuffle()
        self.unshuffle_solution()
        
        return self.x_total, self.f_total