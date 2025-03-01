from mesh import Element, Mesh
import numpy as np

class Beam(Element):
    """Beam class that extends from the element class."""

    def beam_stiffness(self, nodes):
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
    
    def beam_transform(self, g):
        G = np.zeros((12, 12))
        G[0:3, 0:3] = g
        G[3:6, 3:6] = g
        G[6:9, 6:9] = g
        G[9:12, 9:12] = g
        return G

    def beam_shape(self, nodes, steps):
        # Gets nodes
        # Gets L and position of node 1
        # Gets polynomial coefficients for shape function
        node1, node2 = nodes
        L = np.linalg.norm(node2.pos - node1.pos)
        disp = self.beam_transform(self.rotation).T @ np.append(node1.x_sol, node2.x_sol)
        ushape_matrix = np.zeros((4, 12))
        vshape_matrix = np.zeros((4,12))
        wshape_matrix = np.zeros((4,12))
        axial_shape_matrix = np.array([
            [1, 0],
            [-1, 1],
            [0,0],
            [0,0]
        ])
        bending_shape_matrix = np.array([
            [1, 0, 0, 0],
            [0, L, 0, 0],
            [-3, -2*L, 3, -L],
            [2, L, -2, L]
        ])
        ushape_matrix[:, [0, 6]] = axial_shape_matrix
        ushape_coeff = ushape_matrix @ disp
        vshape_matrix[:,1:12:3] = bending_shape_matrix
        vshape_coeff = vshape_matrix @ disp
        wshape_matrix[:,2:12:3] = bending_shape_matrix
        wshape_coeff = wshape_matrix @ disp
        vwshape_coeff = np.append(np.array([vshape_coeff]), np.array([wshape_coeff]), axis=0)
        uvwshape_coeff = np.append(np.array([ushape_coeff]), vwshape_coeff, axis=0)

        return [
            self.rotation @ uvwshape_coeff @ np.array([1, xi, xi**2, xi**3])
            for xi in steps
        ]


    def __init__(self, E: float, A: float, Iy: float, Iz: float, J:float, v: float, y: np.ndarray):
        self.E = E
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.v = v

        update_stiffness = lambda nodes: self.beam_stiffness(nodes)
        shape = lambda nodes, steps: self.beam_shape(nodes, steps=steps)
        super().__init__(update_stiffness, 6, self.beam_transform, y, shape)

