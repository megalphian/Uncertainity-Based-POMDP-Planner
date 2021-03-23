from shapely.geometry import Polygon
import numpy as np

class Environment:

    def __init__(self, rect_limits, sampling_resolution):
        
        self.rect_limits = rect_limits
        self.resolution = sampling_resolution
        self.coords = self.getPolygonBounds()

        self.boundary = Polygon(self.coords)
        
        self.sampled_points = self.sample_environment()
        
        # Holds the uncertainity
        self.uncertainity_distribution = list()
        self.set_uncertainity()
        
        # We could either use an occupancy grid or a list of known obstacles
        self.obstacles = None
        self.occupancy = None
    
    def sample_environment(self):
        x, y = np.meshgrid(np.arange(self.rect_limits[0],self.rect_limits[1],self.resolution), np.arange(self.rect_limits[0],self.rect_limits[1],self.resolution))
        x, y = x.flatten(), y.flatten()
        return np.vstack((x,y)).T

    def getPolygonBounds(self):
        return ((self.rect_limits[0], self.rect_limits[0]),\
            (self.rect_limits[1], self.rect_limits[0]),\
            (self.rect_limits[1], self.rect_limits[1]),\
            (self.rect_limits[0], self.rect_limits[1]))

    def set_uncertainity(self):
        for point in self.sampled_points:
            x = point[0]
            uncertainity_val = 400 - (10 * (x-6)**2)
            uncertainity_val = uncertainity_val if uncertainity_val >= 0 else 0
            self.uncertainity_distribution.append(uncertainity_val)
    
    def take_measurement(self, point):
        # Returns the uncertainity experienced in the point
        uncertainity = self.uncertainity_distribution[point]
        measurement = self.occupancy[point]

        return (measurement, uncertainity)