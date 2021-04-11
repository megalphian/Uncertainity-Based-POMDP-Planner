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
            uncertainity_val = self.calc_uncertainity(x)
            self.uncertainity_distribution.append(uncertainity_val)
    
    def calc_uncertainity(self, x):
        uncertainity_val = ((x - 5) ** 2)
        return uncertainity_val
    
    def get_measurement(self, state):

        uncertainity_sig = self.calc_uncertainity(state[0])
        N = uncertainity_sig * np.identity(2)
        
        n_x = np.random.normal(0, 1)
        n_y = np.random.normal(0, 1)

        n = np.array([[n_x], [n_y]])

        m_x = state[0] + (n_x * uncertainity_sig)
        m_y = state[1] + (n_y * uncertainity_sig)
        measurement = np.array([[m_x], [m_y]])

        return [measurement, N]
