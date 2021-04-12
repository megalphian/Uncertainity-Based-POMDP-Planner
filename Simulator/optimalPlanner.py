import math
import random

import matplotlib.pyplot as plt
import numpy as np

class OptimalPath:

    def __init__(self, path, inputs):
        self.path = path
        self.inputs = inputs
    
    def get_path_length(self):
        return len(self.path)
    
    def get_control_length(self):
        return len(self.inputs)

class StraightLinePlanner:
    def __init__(self, start, common_config, straight_line_planner_config):

        self.start = start
        
        self.goal = common_config.goal
        self.time_step = common_config.time_step

        self.path_resolution = straight_line_planner_config.path_resolution

        self.optimal_path = OptimalPath([start], list())

        self.dist = np.sqrt(((self.goal[1] - self.start[1])**2) + ((self.goal[0] - self.start[0])**2))
        self.steps = round(self.dist/ (self.path_resolution))

        self.sin_angle = np.arcsin((self.goal[1] - start[1])/ self.dist)
        self.cos_angle = np.arccos((self.goal[0] - start[0])/ self.dist)
    
    def generatePath(self):
        
        for i in range(self.steps):
            u_x = self.path_resolution * np.cos(self.cos_angle)
            u_y = self.path_resolution * np.sin(self.sin_angle) 
            x =  u_x + self.optimal_path.path[i][0]
            y =  u_y + self.optimal_path.path[i][1]

            u_x = u_x/self.time_step
            u_y = u_y/self.time_step
            
            self.optimal_path.inputs.append([u_x, u_y])
            self.optimal_path.path.append([x,y])            
        
        return self.optimal_path