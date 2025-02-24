from mesh import Mesh
from beam import Beam
import numpy as np
import pytest

def run_test_x_axial():
    # Tests pure axial loading
    mesh = Mesh(6)
    F = 20
    E = 100
    L = 3
    A = 6
    Iy = 6
    Iz = 3
    J = 2
    v = 0.2

    mesh.add_node(0, np.array([0,0,0]), np.zeros(6))
    mesh.add_node(1, np.array([L,0,0]), np.full(6, np.nan), bf=np.array([F, 0, 0, 0, 0, 0]))

    beam = Beam(E=E, A=A, Iy = Iy, Iz = Iz, J = J, v = v, y=np.array([0, 1, 0]))
    mesh.add_element(beam, 0, 1)

    x, f = mesh.solve()

    d_n1 = np.array([0, 0, 0, 0, 0, 0])
    d_n2 = np.array([F*L/(E*A), 0, 0, 0, 0, 0])
    d = np.append(d_n1, d_n2)

    r_n1 = np.array([-F, 0, 0, 0, 0, 0])
    r_n2 = np.array([0, 0, 0, 0, 0, 0])
    r = np.append(r_n1, r_n2)

    print(x)
    print(f)

    return mesh, x, d, f, r

def run_test_xyz_axial():
    mesh = Mesh(6)
    F = 20
    E = 100
    A = 1
    X = 3
    Y = 2
    Z = 4
    Iy = 1
    Iz = 1
    J = 1
    v = 0.2

    L = np.sqrt(X**2 + Y**2 + Z**2)
    Fx = F*X/L
    Fy = F*Y/L
    Fz = F*Z/L

    mesh.add_node(0, np.array([0,0,0]), np.zeros(6))
    mesh.add_node(1, np.array([X,Y,Z]), np.full(6, np.nan), bf=np.array([Fx, Fy, Fz, 0, 0, 0]))

    beam = Beam(E=E, A=A, Iy = Iy, Iz = Iz, J = J, v = v, y=np.array([0, 1, 0]))
    mesh.add_element(beam, 0, 1)

    x, f = mesh.solve()

    d_n1 = np.array([0, 0, 0, 0, 0, 0])
    d_n2 = np.array([F*X/(E*A), F*Y/(E*A), F*Z/(E*A), 0, 0, 0])
    d = np.append(d_n1, d_n2)

    r_n1 = np.array([-Fx, -Fy, -Fz, 0, 0, 0])
    r_n2 = np.array([0, 0, 0, 0, 0, 0])
    r = np.append(r_n1, r_n2)

    return mesh, x, d, f, r

def run_test_x_cantilever():
    mesh = Mesh(6)
    F = 20
    E = 100
    A = 1
    L = 7
    Iy = 1
    Iz = 1
    J = 1
    v = 0.2

    mesh.add_node(0, np.array([0,0,0]), np.zeros(6))
    mesh.add_node(1, np.array([L,0,0]), np.full(6, np.nan), bf=np.array([0, -F, 0, 0, 0, 0]))

    beam = Beam(E=E, A=A, Iy = Iy, Iz = Iz, J = J, v = v, y=np.array([0, 1, 0]))
    mesh.add_element(beam, 0, 1)

    x, f = mesh.solve()

    d_n1 = np.array([0, 0, 0, 0, 0, 0])
    d_n2 = np.array([0, -F*L**3/(3*E*Iz), 0, 0, 0, -F*L**2/(2*E*Iz)])
    d = np.append(d_n1, d_n2)

    r_n1 = np.array([0, F, 0, 0, 0, F*L])
    r_n2 = np.array([0, 0, 0, 0, 0, 0])
    r = np.append(r_n1, r_n2)

    return mesh, x, d, f, r

def run_test_x_simple_simple():
    mesh = Mesh(6)
    F = 1
    E = 1
    A = 1
    L = 1
    Iy = 1
    Iz = 1
    J = 1
    v = 0.2

    mesh.add_node(0, np.array([0,0,0]), np.array([0, 0, 0, np.nan, np.nan, np.nan]))
    mesh.add_node(1, np.array([L/2, 0, 0]), np.array([np.nan, np.nan, np.nan, 0, 0, 0]), bf=np.array([0, -F, 0, 0, 0, 0]))
    mesh.add_node(2, np.array([L, 0,0]), np.array([0, 0, 0, np.nan, np.nan, np.nan]))

    beam = Beam(E=E, A=A, Iy = Iy, Iz = Iz, J = J, v = v, y=np.array([0, 1, 0]))
    mesh.add_element(beam, 0, 1)
    mesh.add_element(beam, 1, 2)

    x, f = mesh.solve()

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

    return mesh, x, d, f, r