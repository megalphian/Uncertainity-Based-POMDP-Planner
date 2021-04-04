# RRT Planner class
# Source: 
#   Atsushi Sakai et al.
#   https://arxiv.org/abs/1808.10703
#   Code: https://github.com/AtsushiSakai/PythonRobotics

# Changes made by Megnath Ramesh to integrate with project

from stateNode import StateNode

import math
import random

import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import Point

class StraightLinePlanner:
    def __init__(self, start, goal, path_resolution=0.3):

        self.start = start
        self.goal = goal
        self.path_resolution = path_resolution

        self.path = [start]
        self.inputs = list()

        self.dist = np.sqrt(((self.goal[1] - self.start[1])**2) + ((self.goal[0] - self.start[0])**2))
        self.steps = round(self.dist/ self.path_resolution)

        self.sin_angle = np.arcsin((goal[1] - start[1])/ self.dist)
        self.cos_angle = np.arccos((goal[0] - start[0])/ self.dist)
    
    def generatePath(self):
        
        for i in range(self.steps):
            u_x = self.path_resolution * np.cos(self.cos_angle)
            u_y = self.path_resolution * np.sin(self.sin_angle) 
            x =  u_x + self.path[i][0]
            y =  u_y + self.path[i][1]
            self.path.append([x,y])
            self.inputs.append([u_x, u_y])
        
        return (self.path, self.inputs)

class RRTPlanner:
    """
    Class for RRT planning
    """

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=0.5,
                 path_resolution=0.1,
                 goal_sample_rate=5,
                 max_iter=500):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.start = StateNode(start[0], start[1])
        self.end = StateNode(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, animation=False):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].state.x,
                                      self.node_list[-1].state.y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = StateNode(from_node.state.x, from_node.state.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path = [new_node.state]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node_x = self.path_resolution * math.cos(theta) + new_node.state.x
            new_node_y = self.path_resolution * math.sin(theta) + new_node.state.y
            new_node.state = Point(new_node_x, new_node_y)
            new_node.path.append(new_node.state)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path.append(to_node.state)
            new_node.state = Point(to_node.state.x, to_node.state.y)

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.state.x, self.end.state.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.state.x, node.state.y])
            node = node.parent
        path.append([node.state.x, node.state.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.state.x
        dy = y - self.end.state.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = StateNode(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = StateNode(self.end.state.x, self.end.state.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.state.x, rnd.state.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        plt.plot(self.start.state.x, self.start.state.y, "xr")
        plt.plot(self.end.state.x, self.end.state.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.state.x - rnd_node.state.x)**2 + (node.state.y - rnd_node.state.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(node, obstacleList):

        if node is None:
            return False
        
        if(obstacleList is None):
            return True
        
        safe = True

        for obstacle in obstacleList:
            for path_node in node.path:
                if(obstacle.contains(path_node)):
                    safe = False
                    break

        return safe # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.state.x - from_node.state.x
        dy = to_node.state.y - from_node.state.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta
