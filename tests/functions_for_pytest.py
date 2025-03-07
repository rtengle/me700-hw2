from mesh import Mesh
from beam import Beam
import numpy as np
import pytest
import matplotlib.pyplot as plt
from random_utils import *

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

    eigval, eigvec = mesh.global_eigenmode_study(Beam.beam_buckling_eigenmatrix)

    l = eigval[0]
    b = eigvec[:,0]

    d_n1 = np.array([0, 0, 0, 0, 0, 0])
    d_n2 = np.array([F*L/(E*A), 0, 0, 0, 0, 0])
    d = np.append(d_n1, d_n2)

    r_n1 = np.array([-F, 0, 0, 0, 0, 0])
    r_n2 = np.array([0, 0, 0, 0, 0, 0])
    r = np.append(r_n1, r_n2)

    critload = np.pi**2 * E * np.min([Iy, Iz]) / (4*L**2)

    return mesh, x, d, f, r, np.abs(l*F), critload

def run_test_xyz_axial():
    mesh = Mesh(6)
    F = -20
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

    eigval, eigvec = mesh.global_eigenmode_study(Beam.beam_buckling_eigenmatrix)

    l = eigval[0]
    b = eigvec[:,0]

    critload = np.pi**2 * E * np.min([Iy, Iz]) / (4*L**2)

    return mesh, x, d, f, r, np.abs(l*F), critload

def run_test_xyz_discrete():
    # Initialize mesh

    # Initialize mesh

    mesh = Mesh(6)

    N = 30

    x = np.linspace(0, 25, N)
    y = np.linspace(0, 50, N)
    z = np.linspace(0, 37, N)

    L = np.sqrt( x[-1]**2 + y[-1]**2 + z[-1]**2 )

    P = -1

    Fx = P * x[-1]/L
    Fy = P * y[-1]/L
    Fz = P * z[-1]/L

    bf = np.array([Fx, Fy, Fz, 0, 0, 0])

    mesh.add_node(0, np.array([0, 0, 0]), bc=np.zeros(6))
    mesh.add_node(N-1, np.array([x[-1], y[-1], z[-1]]), bc=np.full(6, np.nan), bf=bf)

    for i in range(1, N-1):
        mesh.add_node(
            i, 
            pos=np.array([x[i], y[i], z[i]]),
            bc=np.full(6, np.nan)
        )

    r = 1
    E = 10000
    v = 0.3
    A = np.pi * r**2
    Iy = np.pi * r**4/4
    Iz = Iy
    J = Iy + Iz

    beam = Beam(E=E, A=A, Iy=Iy, Iz=Iz, J=J, v=v, y=np.array([0, 1, 0]))

    for i in range(N-1):
        mesh.add_element(beam, i, i+1)

    d = np.array([P*x[-1]/(E*A), P*y[-1]/(E*A), P*z[-1]/(E*A), 0, 0, 0])
    r = np.array([-Fx, -Fy, -Fz, 0, 0, 0])

    x, f = mesh.solve()

    eigval, eigvec = mesh.global_eigenmode_study(Beam.beam_buckling_eigenmatrix)
    l = eigval[0]
    b = eigvec[:,0]

    critload = np.pi**2 * E * np.min([Iy, Iz]) / (4*L**2)

    return mesh, x[-6:], d, f[0:6], r, np.abs(l*P), critload

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

def run_test_shape():
    mesh = Mesh(6)
    d1 = np.array([3, -7, 4, -0.4, 0.5, 0.4])
    d2 = np.array([-2, 11, 1, 0, 0.7, 0.3])
    L = 2
    mesh.add_node(0, pos=np.array([1, 0, 0]), bc=d1)
    mesh.add_node(1, pos=np.array([1+L, 0, 0]), bc=d2)
    beam = Beam(1432, 2.3, 1.1, 1.3, 1.1+1.3, 0.43, np.array([0, 1, 0]))
    mesh.add_element(beam, 0, 1)
    x, f = mesh.solve()
    n1 = mesh.nodes[0]['object']
    n2 = mesh.nodes[1]['object']
    xi = np.linspace(0, 1, 50)
    s = np.array(beam.beam_shape((n1, n2), xi))
    a1 = d1[1]
    a2 = d1[5]*L
    a3 = (-3*d1[1] + 3*d2[1] - 2*d1[5]*L - d2[5]*L)
    a4 = (2*d1[1] - 2*d2[1] + d1[5]*L + d2[5]*L)
    sy = np.array([a1 + a2*z + a3*z**2 + a4*z**3 for z in xi])
    # Need to invert thetay bc, when analyzing the z-displacement, the relevant 
    # rotation points against the y-axis.
    a1 = d1[2]
    a2 = -d1[4]*L
    a3 = (-3*d1[2] + 3*d2[2] + 2*d1[4]*L + d2[4]*L)
    a4 = (2*d1[2] - 2*d2[2] - d1[4]*L - d2[4]*L)
    sz = np.array([a1 + a2*z + a3*z**2 + a4*z**3 for z in xi])
    sx = np.array([d1[0]*(1-z) + d2[0]*z for z in xi])
    strue = np.array([sx, sy, sz]).T
    return s, strue

def run_test_plot():
    mesh, x, d, f, r, l, critload = run_test_xyz_discrete()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    mesh.plot(ax, disp_scale=10)
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    set_axes_equal(ax)

    eigval, eigvec = mesh.global_eigenmode_study(Beam.beam_buckling_eigenmatrix)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    mesh.plot_displacement(ax, eigvec[:, 0], disp_scale=5, force_scale=10)