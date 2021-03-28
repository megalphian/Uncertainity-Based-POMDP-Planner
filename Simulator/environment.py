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
        uncertainity_val = (- (x + 3) * (x - 16)) / 7.5
        return uncertainity_val
    
    def get_measurements(self, states):
        # Returns the uncertainity experienced in the point
        measurements = list()
        N = list()
        for i in range(len(states) - 1):
            state = states[i+1]
            uncertainity_val = self.calc_uncertainity(state[0])
            print(uncertainity_val)
            n = uncertainity_val # No random gaussian element to the noise. Just gonna be a quadratic function
            N.append(np.asarray([[uncertainity_val]]))
            measurement = state[0] + state[1] + n
            measurements.append(measurement)
        return [measurements, N]