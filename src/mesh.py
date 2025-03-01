from networkx import Graph
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

def unit(v):
    # Adding in a function that should be a default in numpy: getting the unit vector
    return v/np.linalg.norm(v)



class Node():
    """Class representing a Mesh node in a truss system.
    
    Attributes
    ----------

    pos : ndarray
        3D vector representing the position of the node.
    bc : ndarray
        Vector representing the displacement boundary conditions of the node. NaN means it's free along that DoF
    bf : ndarray
        Total external forces acting on that node. Used for solving the system.
    """
    def __init__(self, pos: np.ndarray, bc: np.ndarray, bf = None):
        # Initializes node using a position vector with a set of displacement boundary conditions. NaN is considered free.
        self.pos = pos
        self.bc = bc
        if type(bf) != np.ndarray and bf == None:
            self.bf = np.zeros(len(bc))
        else:
            self.bf = bf

    def add_bf(self, bf):
        """Adds a body force contribution to the node.

        Attributes
        ----------

        bf : ndarray
            Array representing the force contributions along the DoF
        """
        self.bf = self.bf + bf

    def set_bc(self, bc: np.ndarray):
        """Sets the total displacement boundary conditions in the node
        
        Attributes
        ----------

        bc : ndarray
            Vector represention of the displacement boundary. NaN is free along that DoF
        """
        self.bc = bc

    def add_bc(self, bc: type[float | np.ndarray], dof: type[int | slice]):
        """Sets the boundary condition at a specific DoF
        
        Attributes
        ----------

        bc : float, ndarray
            Boundary condition value. NaN means it's free in that DoF
        dof : int, slice
            DoF index or set of DoFs that are being changed.
        """
        self.bc[dof] = bc

    def set_solution(self, x, f):
        """Sets the solved displacements and forces for a given node."""
        self.x_sol = x
        self.f_sol = self.bf + f



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
    def __init__(self, stiffness: Callable, body_forces: type[int | Callable], transform: Callable, orientation: np.ndarray, shape_function: Callable):
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
        """
        # Initializes element of mesh with a stiffness method and body method that are able to be dynamically updated along with an orientation vector.
        self.stiffness_method = stiffness
        self.body_forces_method = body_forces
        self.transformation_method = transform
        self.orientation = unit(orientation)
        self.shape_function = shape_function

    def reorient(self, xvec: np.ndarray, output:bool=True):
        """Function that sets the rotation matrix transforming local --> global.
        
        Attributes
        ----------

        xhat : ndarray
            3D unit vector representing the length of the element
        output : bool
            Boolean that specifies if the rotation should be returned

        Returns
        -------

        self.rotation : ndarray
            3x3 rotation matrix transforming the local coordinates to the global coordinates
        """
        # Calculates the unit vector of x
        xhat = unit(xvec)
        # Gets the z vector from crossing the x and xy vectors
        zhat = unit(np.cross(xvec, self.orientation))
        # Gets the y  vector from the z and x vector 
        yhat = np.cross(zhat, xhat)
        # Sets the rotation internally and outputs the results if desired
        self.rotation = np.array([xhat, yhat, zhat]).T
        if output == True:
            return self.rotation

    def local_stiffness(self, nodes: tuple, state = None, output:bool=True):
        """Function that automatically calculates the local stiffness matrix based on the state and nodes based on a specified method.

        Attributes
        ----------
        
        nodes : (Node, Node)
            Tuple of Node objects representing the start and end point of the edge. 
        state : None, dict
            State dictionary containing a set of data for the element to calculate the stiffness matrix
        output : bool
            Boolean that specifies if the local stiffness matrix should be returned

        Returns
        -------
        
        self.local_stiffness_matrix : ndarray
            Local stiffness matrix in the element's local coordinates
        """
        # Just calls the method assigned in the contructor
        if state == None:
            self.local_stiffness_matrix = self.stiffness_method(nodes)
        else:
            self.local_stiffness_matrix = self.stiffness_method(nodes, state)
        if output == True:
            return self.local_stiffness_matrix
    
    def local_body_forces(self, nodes, state=None, output=True):
        """Calculates the local body forces for a set of nodes and a state
        
        Attributes
        ----------
        
        nodes : (Node, Node)
            Tuple of Node objects representing the start and end point of the edge. 
        state : None, dict
            State dictionary containing a set of data for the element to calculate the stiffness matrix
        output : bool
            Boolean that specifies if the local body force contributions should be returned

        Returns
        -------
        
        self.local_stiffness_matrix : ndarray
            Local body force contributions in the element's local coordinates
        """
        # Just calls the method assigned in the constructor. If it's an int, it just creates an empty 1D vector
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
    
    def global_transform(self, nodes: tuple, output=True):
        """Creates the global transformation matrix that transitions from the local DoF frame to the global DoF frame based on a rotation matrix:
        This needs to be its own matrix because there are multiple sets of transforms and some DoFs don't such as temperature or charge.

        Attributes
        ----------
        
        nodes : tuple
            Tuple of nodes used to get the x-vector when passing to the reorient method
        output : bool
            Boolean used to determine if the full transform should be returned

        Returns
        -------
        
        self.global_transformation_matrix : ndarray
            Matrix that transforms the local element DoFs to the global DoF frame.
        """
        # Gets the x-vector
        node1, node2 = nodes
        L = node2.pos - node1.pos
        # Gets the global transformation matrix and returns it if needed
        self.global_transformation_matrix = self.transformation_method(self.reorient(L))
        if output == True:
            return self.global_transformation_matrix
        
    def global_coords(self, nodes: tuple, state=None):
        """Gets the stiffness matrix and body force for the element in the global reference frame.
        
        Attributes
        ----------
        nodes : tuple
            Tuple containing the connecting nodes
        state
            Set of data relevant in calculating the forces and stiffness matrix

        Returns
        -------
        self.global_stiffness_matrix : ndarray
            Stiffness matrix of the element in the global reference frame
        self.global_body_forces : tuple
            Tuple of the body force contributions from the node in the global reference frame
        """
        # Gets the global coordinate values for the stiffness and body forces
        K = self.local_stiffness(nodes, state=state)
        fb1, fb2 = self.local_body_forces(nodes, state=state)
        fb = np.append(fb1, fb2)
        G = self.global_transform(nodes)
        self.global_stiffness_matrix = G @ K @ G.T
        global_fb = G @ fb
        fg1, fg2 = np.split(global_fb, 2)
        self.global_body_forces = (fg1, fg2)
        return self.global_stiffness_matrix, self.global_body_forces



class Mesh(Graph):
    #TODO: Finish up docstrings and comments apart from basic functionality
    """Class of a mesh containing nodes connected by elements. These elements act as subdomains of the system while the nodes act as subboundaries.
    These are connected through linear relationships of the form f = Kx with a set of prescribed f and x.
    """
    def __init__(self, dof):
        self.node_dof = dof
        super().__init__()

    def add_node(self, id: int, pos: np.ndarray, bc=None, bf=None):
        """Adds a node with an integer identifier, a position vector, a set of boundary conditions, and set of forces
        """
        if type(bc) != np.ndarray and bc == None:
            bc = self.dof
        super().add_node(node_for_adding=id, object=Node(pos=pos, bc=bc, bf=bf))

    def add_element(self, element: Element, n1, n2):
        """Adds an element connected by two nodes. Direction is n1 --> n2
        """
        super().add_edge(n1, n2, object=element)

    def add_stiffness(self, K_local, n1, n2):
        """Adds the local stiffness matrix (in the global coordinates) of element connected by node indicies n1 and n2 
        to the total global stiffness matrix.
        """
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
        """Adds forces on a node with index i to the total global external forces vector"""
        self.external_forces[i*self.node_dof:(i+1)*self.node_dof] += node.bf

    def set_boundary_conditions(self, i, node: Node):
        """Specifies the boundary conditions of a given node in the total boundary condition vector"""
        self.total_bc[self.node_dof*i:(i+1)*self.node_dof] = node.bc

    def iterate_nodes(self):
        """Iterates of the nodes to add force and boundary conditions to the total vectors"""
        for (i, node) in self.nodes.data('object'):
            self.add_forces(i, node)
            self.set_boundary_conditions(i, node)

    def get_shuffle_matrix(self):
        """Gets the shuffle matrix describing how the system is shuffled around to separate free DoFs from fixed DoFs"""
        # Sort the boundary conditions in the form [float, ..., nan, ...] and returns the indexing that does this shuffling.
        # In this form, the fixed DoFs are given first and the free DoFs are given last
        bc_sorted = np.argsort(self.total_bc)
        # Initializes shuffle matrix
        self.shuffle_matrix = np.zeros((self.total_dof, self.total_dof))
        # Iterates through the sorted indexes and constructs the shuffle matrix
        for end_index, start_index in zip(range(self.total_dof), bc_sorted):
            self.shuffle_matrix[end_index, start_index] = 1
        # Gets the shuffled boundary conditions
        self.bc_shuffle = self.total_bc[bc_sorted]

    def shuffle_system(self):
        """Shuffles the system so that the free DoFs are on the top and fixed DoFs are on the bottom"""
        # Gets the matrix describing how the system is shuffled. These are a special class of rotation matrices
        self.get_shuffle_matrix()
        # Shuffles the K matrix to the new set of systems.
        self.K_shuffle = self.shuffle_matrix @ self.K_total @ self.shuffle_matrix.T
        # Shuffles the forces
        self.shuffled_forces = self.shuffle_matrix @ self.external_forces
        

    def assemble_global(self):
        """Assembles the total global stiffness matrix for the full mesh system"""
        # Iterate through edges, update them, grab their stiffness matrices, and sum them

        # Initializing important variables 
        self.total_dof = self.node_dof*self.number_of_nodes()
        self.K_total = np.zeros((self.total_dof, self.total_dof))
        self.external_forces = np.zeros(self.total_dof)
        self.total_bc = np.full(self.total_dof, np.nan)

        # Iteration methods
        self.iterate_elements()
        self.iterate_nodes()

    def solve_shuffle(self):
        """Solves the shuffled system of equations"""
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
        """Undoes the shuffled solution"""
        self.x_total = self.shuffle_matrix.T @ self.x_shuffle
        self.f_total = self.shuffle_matrix.T @ self.f_shuffle

    def assign_solution(self):
        """Assigns the solved forces and displacements to their respective nodes"""
        for (n, node) in self.nodes.data('object'):
            x = self.x_total[n*self.node_dof:(n+1)*self.node_dof]
            f = self.f_total[n*self.node_dof:(n+1)*self.node_dof]
            node.set_solution(x, f)

    def solve(self):
        """Solves the system after defining"""
        # Assembles global matrix
        self.assemble_global()
        # Shuffles the system
        self.shuffle_system()
        # Solves the system
        self.solve_shuffle()
        # Unshuffles the system
        self.unshuffle_solution()
        # Assigns the solved forces and displacements to the nodes
        self.assign_solution()
        
        return self.x_total, self.f_total
    
    def plot(self, ax, disp_scale=1, force_scale=1, node_format = 'b', element_format = 'b', shape=True, xi_steps = 50):
        """Plots the entire mesh using the shape functions. General plotting format:
        
        - Nodes are points
        - Elements are lines w/ their normalized orientation vector at the center
        - Shape functions are used if enabled (default yes)
        """
        for (n, node) in self.nodes.data('object'):
            ax.scatter(*node.pos, marker='o', color='k')
            ax.quiver(*node.pos, *(force_scale*unit(node.bf[0:3])), color='k')
            ax.quiver(*node.pos, *(force_scale*unit(node.bf[3:6])), linestyle='dashed', color='k')
            ax.text(*node.pos, "Node {n}", color='red')
        
        for (n1, n2, el) in self.edges.data('object'):
            xi_list = np.linspace(0, 1, xi_steps)
            node1 = self.nodes[n1]['object']
            node2 = self.nodes[n2]['object']
            node_tuple = (node1, node2)
            displacements = el.shape_function(node_tuple, xi_list)
            origins = [node1.pos*(1-xi) + node2.pos*xi for xi in xi_list]
            new_shapes = np.array([x + (u*disp_scale) for x, u in zip(origins, displacements)])
            origins_array = np.array(origins)
            ax.plot(origins_array[:,0], origins_array[:, 1], origins_array[:, 2], 'k--')
            ax.plot(new_shapes[:,0], new_shapes[:,1], new_shapes[:,2], 'k')
            ax.text(*(node1.pos + node2.pos)/2, "El. ({n1}, {n2})", color='green')
            

