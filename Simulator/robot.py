from shapely.geometry import Point

class Robot:
    def __init__(self, init_state):
        
        self.init_state = init_state

        # Define the belief dynamics of the robot
        self.state_belief = self.init_state # type Point
        self.state_covariance = None
        self.state_measurement = None

        # Matrices to scale the process and measurement noises
        self.M = None
        self.N = None
    
    def update_state(self, control_input, time_duration, process_noise):
        self.state += control_input * time_duration

    def update_measurement(self, measurement):
        self.state_measurement = measurement