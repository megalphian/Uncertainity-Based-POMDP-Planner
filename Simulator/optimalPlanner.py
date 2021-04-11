from stateNode import StateNode

import math
import random

import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import Point

class StraightLinePlanner:
    def __init__(self, start, goal, path_resolution=0.3, time_step=1):

        self.start = start
        self.goal = goal
        self.path_resolution = path_resolution
        self.time_step = time_step

        self.path = [start]
        self.inputs = list()

        self.dist = np.sqrt(((self.goal[1] - self.start[1])**2) + ((self.goal[0] - self.start[0])**2))
        self.steps = round(self.dist/ (self.path_resolution))

        self.sin_angle = np.arcsin((goal[1] - start[1])/ self.dist)
        self.cos_angle = np.arccos((goal[0] - start[0])/ self.dist)
    
    def generatePath(self):
        
        for i in range(self.steps):
            u_x = self.path_resolution * np.cos(self.cos_angle)
            u_y = self.path_resolution * np.sin(self.sin_angle) 
            x =  u_x + self.path[i][0]
            y =  u_y + self.path[i][1]

            u_x = u_x/self.time_step
            u_y = u_y/self.time_step
            
            self.inputs.append([u_x, u_y])
            self.path.append([x,y])            
        
        return (self.path, self.inputs)