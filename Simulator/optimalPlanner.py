import math
import random

import matplotlib.pyplot as plt
import numpy as np

class StartPoint:

    def __init__(self, start_hat, init_cov_val):

        self.start_hat = start_hat
        self.init_cov_mat = init_cov_val * np.identity(2)
        
class OptimalPath:

    def __init__(self, path, inputs, start_point, goal):
        self.path = path
        self.inputs = inputs
        self.start_point = start_point
        self.goal = goal
    
    def get_path_length(self):
        return len(self.path)
    
    def get_control_length(self):
        return len(self.inputs)

class StraightLinePlanner:
    def __init__(self, common_config, straight_line_planner_config):

        # Initial starting belief
        start_hat = common_config.start_hat
        cov_val = common_config.init_cov_val

        start_point = StartPoint(start_hat, cov_val)
        
        goal = common_config.goal
        self.time_step = common_config.time_step

        self.velocity = straight_line_planner_config.velocity
        self.step_length = self.velocity * self.time_step

        self.optimal_path = OptimalPath([start_hat], list(), start_point, goal)

        self.dist = np.sqrt(((goal[1] - start_hat[1])**2) + ((goal[0] - start_hat[0])**2))
        self.steps = round(self.dist/ (self.step_length))

        self.sin_angle = np.arcsin((goal[1] - start_hat[1])/ self.dist)
        self.cos_angle = np.arccos((goal[0] - start_hat[0])/ self.dist)
    
    def generatePath(self):
        
        for i in range(self.steps):
            u_x = self.step_length * np.cos(self.cos_angle)
            u_y = self.step_length * np.sin(self.sin_angle) 
            x =  u_x + self.optimal_path.path[i][0]
            y =  u_y + self.optimal_path.path[i][1]

            u_x = u_x/self.time_step
            u_y = u_y/self.time_step
            
            self.optimal_path.inputs.append([u_x, u_y])
            self.optimal_path.path.append([x,y])            
        
        return self.optimal_path