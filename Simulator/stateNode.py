from shapely.geometry import Point

class StateNode:

    def __init__(self, x, y):
        self.state = Point(x,y)
        self.path_x = []
        self.path_y = []
        self.parent = None