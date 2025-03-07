from mesh import Element, Node
import numpy as np

class Beam(Element):
    """Beam class that extends from the element class. This class specifically represends a slender beam
    obeying Euler-Bernoulli beam theory.
    """

    def beam_stiffness(self, nodes: tuple):
        """Returns the elastic stiffness matrix for the beam.
        
        Attributes
        ----------
        nodes : tuple
            Tuple of node objects in the form (node1, node2).

        Returns
        -------
        Ke : ndarray
            Elastic stiffness matrix for the element in local coordinates. 
        """
        node1, node2 = nodes
        L = np.linalg.norm(node2.pos - node1.pos)
        K = np.zeros((12, 12))
        K11 = np.array([
            [self.A/L,  0,                  0,                  0,                          0,                  0],
            [0,         12*self.Iz/L**3,    0,                  0,                          0,                  6*self.Iz/L**2],
            [0,         0,                  12*self.Iy/L**3,    0,                          -6*self.Iy/L**2,    0],
            [0,         0,                  0,                  self.J/(2*(1+self.v)*L),    0,                  0],
            [0,         0,                  -6*self.Iy/L**2,    0,                          4*self.Iy/L,        0],
            [0,         6*self.Iz/L**2,     0,                  0,                          0,                  4*self.Iz/L]
        ])
        K12 = np.array([
            [-self.A/L, 0,                  0,                  0,                          0,                  0],
            [0,         -12*self.Iz/L**3,   0,                  0,                          0,                  6*self.Iz/L**2],
            [0,         0,                  -12*self.Iy/L**3,   0,                          -6*self.Iy/L**2,    0],
            [0,         0,                  0,                  -self.J/(2*(1+self.v)*L),   0,                  0],
            [0,         0,                  6*self.Iy/L**2,     0,                          2*self.Iy/L,        0],
            [0,         -6*self.Iz/L**2,    0,                  0,                          0,                  2*self.Iz/L]
        ])
        K21 = np.array([
            [-self.A/L, 0,                  0,                  0,                          0,                  0],
            [0,         -12*self.Iz/L**3,   0,                  0,                          0,                  -6*self.Iz/L**2],
            [0,         0,                  -12*self.Iy/L**3,   0,                          6*self.Iy/L**2,     0],
            [0,         0,                  0,                  -self.J/(2*(1+self.v)*L),   0,                  0],
            [0,         0,                  -6*self.Iy/L**2,    0,                          2*self.Iy/L,        0],
            [0,         6*self.Iz/L**2,     0,                  0,                          0,                  2*self.Iz/L]
        ])
        K22 = np.array([
            [self.A/L,  0,                  0,                  0,                          0,                  0],
            [0,         12*self.Iz/L**3,    0,                  0,                          0,                  -6*self.Iz/L**2],
            [0,         0,                  12*self.Iy/L**3,    0,                          6*self.Iy/L**2,     0],
            [0,         0,                  0,                  self.J/(2*(1+self.v)*L),    0,                  0],
            [0,         0,                  6*self.Iy/L**2,     0,                          4*self.Iy/L,        0],
            [0,         -6*self.Iz/L**2,     0,                 0,                          0,                  4*self.Iz/L]
        ])
        K[0:6, 0:6] = K11 # top left
        K[6:12, 6:12] = K22 # bottom left
        K[0:6, 6:12] = K12 # top right
        K[6:12, 0:6] = K21 # bottom left
        return self.E*K
    
    def beam_transform(self, g:np.ndarray):
        """Returns the global transformation matrix of the element using a local transformation matrix g.
        This isn't general because not all DoFs transform with rotation (ex: Temperature).

        Attributes
        ----------
        g : ndarray
            3D Rotation Matrix describing a transformation from global coordinates to local coordinates.

        Returns
        -------
        G : ndarray
            Rotation Matrix describing a transformation from the DoFs in the global reference frame to
            the DoFs in the local reference frame.
        """
        G = np.zeros((12, 12))
        G[0:3, 0:3] = g
        G[3:6, 3:6] = g
        G[6:9, 6:9] = g
        G[9:12, 9:12] = g
        return G

    def beam_shape(self, nodes, steps, disp=None):
        """Returns the shape function as parameterized by steps"""
        # Gets nodes
        # Gets L and position of node 1
        # Gets polynomial coefficients for shape function
        node1, node2 = nodes
        L = np.linalg.norm(node2.pos - node1.pos)
        if type(disp) == type(None):
            disp = self.beam_transform(self.rotation).T @ np.append(node1.x_sol, node2.x_sol)
        else:
            disp = self.beam_transform(self.rotation).T @ disp
        # For some reason I thought it would be a good idea to turns the shape functions into matrix multiplication
        # I don't exactly remember why, maybe I thought it would be cleaner but I'm sure sure if it is. Oh well.
        
        # Initializes shape matrices
        ushape_matrix = np.zeros((4,12))
        vshape_matrix = np.zeros((4,12))
        wshape_matrix = np.zeros((4,12))
        # Assigns shape matrices of isolated effects
        axial_shape_matrix = np.array([
            [1, 0],
            [-1, 1],
            [0,0],
            [0,0]
        ])
        vbending_shape_matrix = np.array([
            [1, 0, 0, 0],
            [0, L, 0, 0],
            [-3, -2*L, 3, -L],
            [2, L, -2, L]
        ])
        wbending_shape_matrix = np.array([
            [1, 0, 0, 0],
            [0, -L, 0, 0],
            [-3, 2*L, 3, L],
            [2, -L, -2, -L]
        ])
        # Gets the coefficients along each axis
        ushape_matrix[:, [0, 6]] = axial_shape_matrix
        ushape_coeff = ushape_matrix @ disp
        vshape_matrix[:,[1, 5, 7, 11]] = vbending_shape_matrix
        vshape_coeff = vshape_matrix @ disp
        wshape_matrix[:,[2, 4, 8, 10]] = wbending_shape_matrix
        wshape_coeff = wshape_matrix @ disp
        # Assembles coefficients into a matrix.
        vwshape_coeff = np.append(np.array([vshape_coeff]), np.array([wshape_coeff]), axis=0)
        uvwshape_coeff = np.append(np.array([ushape_coeff]), vwshape_coeff, axis=0)

        # Returns an array of the shape functions
        return np.array([
            self.rotation @ uvwshape_coeff @ np.array([1, xi, xi**2, xi**3])
            for xi in steps
        ])

    @staticmethod
    def beam_buckling_eigenmatrix(nodes: tuple, el: Element):
        """Eigenmatrix for beam buckling.
        
        Attributes
        ----------
        nodes : tuple
            Tuple of the nodes associated with the element el.
        el : Element
            Element to get the matrices for. Should be Beam but I can't define an input as the class it's in.

        Returns
        -------
        A : ndarray
            Elastic stiffness matrix 
        B : ndarray
            Geometric stiffness matrix
        """
        node1, node2 = nodes
        L = np.linalg.norm(node2.pos - node1.pos)
        K = el.beam_stiffness(nodes)
        G = el.beam_transform(el.rotation)
        f = G.T @ el.internal_forces

        My1 = f[4]
        Mz1 = f[5]

        Fx2 = f[6]

        Mx2 = f[9]
        My2 = f[10]
        Mz2 = f[11]

        Ip = el.J
        A = el.A

        Kg = np.zeros((12, 12))
        Kg[0, [0, 6]] = [Fx2/L, -Fx2/L]
        Kg[1, [1, 3, 4, 5, 7, 9, 10, 11]] = [
            6*Fx2/(5*L),
            My1/L,
            Mx2/L,
            Fx2/10,
            -6*Fx2/(5*L),
            My2/L,
            -Mx2/L,
            Fx2/10
        ]
        Kg[2, [2, 3, 4, 5, 8, 9, 10, 11]] = [
            6*Fx2/(5*L),
            Mz1/L,
            -Fx2/10,
            Mx2/L,
            -6*Fx2/(5*L),
            Mz2/L,
            -Fx2/10,
            -Mx2/L
        ]
        Kg[3, [3, 4, 5, 7, 8, 9, 10, 11]] = [
            Fx2*Ip/(A*L),
            -(2*Mz1 - Mz2)/6,
            (2*My1 - My2)/6,
            -My1/L,
            -Mz1/L,
            -(Fx2*Ip)/(A*L),
            -(Mz1+Mz2)/6,
            (My1+My2)/6
        ]
        Kg[4, [4, 7, 8, 9, 10, 11]] = [
            2*Fx2*L/15,
            -Mx2/L,
            Fx2/10,
            -(Mz1+Mz2)/6,
            -Fx2*L/30,
            Mx2/2
        ]
        Kg[5, [5, 7, 8, 9, 10, 11]] = [
            2*Fx2*L/15,
            -Fx2/10,
            -Mx2/L,
            (My1+My2)/6,
            -Mx2/2,
            -Fx2*L/30
        ]
        Kg[6, 6] = Fx2/L
        Kg[7, [7, 9, 10, 11]] = [
            6*Fx2/(5*L),
            -My2/L,
            Mx2/L,
            -Fx2/10
        ]
        Kg[8, [8, 9, 10, 11]] = [
            6*Fx2/(5*L),
            -Mz2/L,
            Fx2/10,
            Mx2/L
        ]
        Kg[9, [9, 10, 11]] = [
            Fx2*Ip/(A*L),
            (Mz1-2*Mz2)/6,
            -(My1-2*My2)/6
        ]
        Kg[10, 10] = 2*Fx2*L/15
        Kg[11, 11] = 2*Fx2*L/15

        Kg = np.triu(Kg, k=1).T + Kg

        return G.T @ K @ G, -G.T @ Kg @ G

    def __init__(self, E: float, A: float, Iy: float, Iz: float, J:float, v: float, y: np.ndarray):
        self.E = E
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.v = v

        update_stiffness = lambda nodes: self.beam_stiffness(nodes)
        shape = lambda nodes, steps, disp=None: self.beam_shape(nodes, steps=steps, disp=disp)
        super().__init__(update_stiffness, 6, self.beam_transform, y, shape)

