from shapely.geometry import Point
import numpy as np

class StateNode:

    def __init__(self, x, y):
        self.state = Point(x,y)
        self.path = []
        self.parent = None

def init_system_matrices(iterations):
    A = list()
    C = list()
    M = list()
    N = list()
    for i in range(iterations):
        A.append(np.identity(2))
        M.append(np.identity(2))
        C.append(np.ones((1,2)))
        N.append(1)
    return [A, C, M, N]