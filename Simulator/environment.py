import numpy as np

class Environment:

    """
    Class to store information about the environment and its uncertainity
    The bounds of the environment are only for illustration.
    The robot takes measurements from the environment, which tells the robot where it is and the state-based uncertainity
    """

    def __init__(self, environment_config):
        
        self.rect_limits = environment_config.env_limits
        self.resolution = environment_config.viz_resolution
        self.light_coord = environment_config.light_coord_x
        
        self.coords = self.getPolygonBounds()
        
        self.sampled_points = self.sample_environment()
        
        # Holds the uncertainity
        self.uncertainity_distribution = self.set_uncertainity()
    
    def sample_environment(self):

        # Sample of the environment for illustration

        x, y = np.meshgrid(np.arange(self.rect_limits[0],self.rect_limits[1],self.resolution), np.arange(self.rect_limits[0],self.rect_limits[1],self.resolution))
        x, y = x.flatten(), y.flatten()
        return np.vstack((x,y)).T

    def getPolygonBounds(self):

        # Get bounds of the environment for illustration

        return ((self.rect_limits[0], self.rect_limits[0]),\
            (self.rect_limits[1], self.rect_limits[0]),\
            (self.rect_limits[1], self.rect_limits[1]),\
            (self.rect_limits[0], self.rect_limits[1]))

    def set_uncertainity(self):

        # Set uncertainity for each sampled point for illustration

        uncertainity_distribution = list()

        for point in self.sampled_points:
            x = point[0]
            y = point[1]
            uncertainity_val = self.calc_uncertainity(x)
            uncertainity_distribution.append(uncertainity_val)

        return uncertainity_distribution
    
    def calc_uncertainity(self, x):

        # Given an x coordinate, calculate the uncertainity distribution given by the quadratic model at that coordinate

        uncertainity_val = ((x - self.light_coord) ** 2) # Adopted from Platt et. al. 2012
        return uncertainity_val
    
    def get_measurement(self, state):

        # Get a measurement at a given robot state. The measurement also returns the noise covariance computed 
        # using the state-dependent uncertainity model

        uncertainity_sig = self.calc_uncertainity(state[0])
        N = uncertainity_sig * np.identity(2)
        
        n_x = np.random.normal(0, 1)
        n_y = np.random.normal(0, 1)

        n = np.array([[n_x], [n_y]])

        m_x = state[0] + (n_x * uncertainity_sig)
        m_y = state[1] + (n_y * uncertainity_sig)
        measurement = np.array([[m_x], [m_y]])

        return [measurement, N]
