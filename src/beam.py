from mesh import Element
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



    def __init__(self, E: float, A: float, Iy: float, Iz: float, J:float, v: float, y: np.ndarray):
        self.E = E
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.v = v

        update_stiffness = lambda nodes: self.beam_stiffness(nodes)
        super().__init__(update_stiffness, 6, self.beam_transform, y)