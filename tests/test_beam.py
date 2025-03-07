import numpy as np
from functions_for_pytest import *
import pytest

def test_x_axial():
    mesh, x, d, f, r, l, critload = run_test_x_axial()
    assert np.linalg.norm(d - x) <= 1e-6
    assert np.linalg.norm(r - f) <= 1e-6
    assert np.linalg.norm(l - critload) <= 1

def test_xyz_axial():
    mesh, x, d, f, r, l, critload = run_test_xyz_axial()
    assert np.linalg.norm(d - x) <= 1e-6
    assert np.linalg.norm(r - f) <= 1e-6
    assert np.linalg.norm(l - critload) <= 1

def test_xyz_discrete():
    mesh, x, d, f, r, l, critload = run_test_xyz_discrete()
    assert np.linalg.norm(d - x) <= 1e-6
    assert np.linalg.norm(r - f) <= 1e-6
    assert np.linalg.norm(l - critload) <= 1e-6

def test_x_cantilever():
    mesh, x, d, f, r = run_test_x_cantilever()
    assert np.linalg.norm(d - x) <= 1e-6
    assert np.linalg.norm(r - f) <= 1e-6

def test_x_simple_simple():
    mesh, x, d, f, r = run_test_x_simple_simple()
    assert np.linalg.norm(d - x) <= 1e-6
    assert np.linalg.norm(r - f) <= 1e-6

def test_shape():
    s_func, s_true = run_test_shape()
    assert np.linalg.norm(s_func - s_true) <= 1e-6

def test_plot():
    run_test_plot()
    pass
