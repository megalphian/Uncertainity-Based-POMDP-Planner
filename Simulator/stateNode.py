from shapely.geometry import Point

class StateNode:

    def __init__(self, x, y):
        self.state = Point(x,y)
        self.path = []
        self.parent = None