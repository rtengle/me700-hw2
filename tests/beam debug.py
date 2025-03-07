import numpy as np
from functions_for_pytest import *
import matplotlib.pyplot as plt

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

mesh = Mesh(6)

F = np.array([-1, 0, 0])
M = np.array([0, 0, 0])
bfb = np.append(F, M)

print(bfb)

mesh.add_node(0, pos=np.array([0, 0, 0]), bc=np.zeros(6))
mesh.add_node(1, pos=np.array([10, 0, 0]), bc=np.full(6, np.nan))
mesh.add_node(2, pos=np.array([20, 0, 0]), bc=np.full(6, np.nan), bf=bfb)

r = 1
A = np.pi * r**2
Iz = np.pi * r**4 / 4
Iy = np.pi * r**4 / 4
J = Iy + Iz
E = 1000
v = 0.3

beam = Beam(E=E, A=A, Iy=Iy, Iz=Iz, J=J, v=v, y=np.array([0, 1, 0]))

mesh.add_element(beam, 0, 1)
mesh.add_element(beam, 1, 2)

x, f = mesh.solve()

val, vec = mesh.global_eigenmode_study(Beam.beam_buckling_eigenmatrix)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
mesh.plot(ax, force_scale=10)
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Z-Axis')
set_axes_equal(ax)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
mesh.plot_displacement(ax, vec[:, 0], disp_scale=5, force_scale=10)
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Z-Axis')
set_axes_equal(ax)
plt.show()