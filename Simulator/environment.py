class Environment:

    def __init__(self):
        # Holds the uncertainity
        self.uncertainity_distribution = None
        
        # We could either use an occupancy grid or a list of known obstacles
        self.obstacles = None
        self.occupancy = None
    
    def take_measurement(self, point):
        # Returns the uncertainity experienced in the point
        uncertainity = self.uncertainity_distribution[point]
        measurement = self.occupancy[point]

        return (measurement, uncertainity)