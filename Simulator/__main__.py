from environment import Environment
from optimalPlanner import StraightLinePlanner
from localUncertainityBasedPlanner import LocalUncertainityBasedPlanner
from config_reader import ConfigParser

import argparse
import os

from matplotlib import pyplot as plt
import numpy as np

# Parse input config file

parser = argparse.ArgumentParser(description='Config file for simulation')
parser.add_argument('--config', type=str, help='path of config file to override default')

dir_path = os.path.dirname(os.path.realpath(__file__))
config_file_path = dir_path + '/../config1.json'

if __name__ == "__main__":

    args = parser.parse_args()
    if args.config:
        config_file_path = args.config

    # Parse simulator config
    config = ConfigParser(config_file_path)

    # Configure the environment and the straight-line planner for the start and goal points
    env = Environment(config.environment_config)
    opt_planner = StraightLinePlanner(config.common_config, config.straight_line_planner_config)
    optimal_path = opt_planner.generatePath()

    # Configure and optimize the straight-line path to a locally optimal path that minimizes uncertainity along the way
    planner = LocalUncertainityBasedPlanner(optimal_path, env, config.common_config, config.cost_function_config)
    planner.optimize_plan_with_uncertainity()

    # Create the plot to show the optimal path, EKF estimations and the uncertainity minimizing path

    fig, ax = plt.subplots()

    for coord in env.coords:
        ax.plot(coord[0], coord[1])

    x, y = zip(*env.sampled_points)
    ax.scatter(x, y, c=env.uncertainity_distribution, cmap='winter_r')

    ax.text(optimal_path.start_point.start_hat[0], optimal_path.start_point.start_hat[1], 'Start Point')
    ax.text(optimal_path.goal[0], optimal_path.goal[1], 'Goal')

    ax.plot([x for (x, y) in optimal_path.path], [y for (x, y) in optimal_path.path], '.-', color='#083D77', label='Optimal Planned Path')
    ax.plot([x for (x, y) in planner.initial_x_est], [y for (x, y) in planner.initial_x_est], '.-.', color='#D00000', label='EKF Estimated Path')
    ax.plot([x for (x, y, _, _, _, _) in planner.current_path], [y for (x, y, _, _, _, _) in planner.current_path], '-o', label='Uncertainity-Minimizing Path')

    ax.legend()

    # Create a plot to show the uncertainity distribution across the environment

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    ax1.scatter(x,y,env.uncertainity_distribution)
    ax1.set_title('Uncertainity Distribution')
    ax1.set_xlabel('X coordinate of robot position')
    ax1.set_ylabel('Uncertainity value')

    plt.show()
