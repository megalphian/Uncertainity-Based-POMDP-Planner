from environment import Environment
from optimalPlanner import StraightLinePlanner
from localUncertainityBasedPlanner import LocalUncertainityBasedPlanner

from matplotlib import pyplot as plt

from config_reader import ConfigParser

import numpy as np

filename = 'config1.json'
config = ConfigParser(filename)
env = Environment(config.environment_config)

cov_val = config.common_config.init_cov_val
init_cov_mat = cov_val * np.identity(2)
# Initial starting belief
start_hat = config.common_config.start_hat

opt_planner = StraightLinePlanner(start_hat, config.common_config, config.straight_line_planner_config)
optimal_path = opt_planner.generatePath()

planner = LocalUncertainityBasedPlanner(optimal_path, env, start_hat, config, init_cov_mat)
planner.optimize_plan_with_uncertainity()

fig, ax = plt.subplots()

for coord in env.coords:
    ax.plot(coord[0], coord[1])

x, y = zip(*env.sampled_points)
ax.scatter(x, y, c=env.uncertainity_distribution, cmap='winter_r')

ax.plot([x for (x, y) in optimal_path.path], [y for (x, y) in optimal_path.path], '.-', color='#083D77')
ax.plot([x for (x, y) in planner.initial_x_est], [y for (x, y) in planner.initial_x_est], '.-.', color='#D00000')
ax.plot([x for (x, y, _, _, _, _) in planner.current_path], [y for (x, y, _, _, _, _) in planner.current_path], '--o')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(x,y,env.uncertainity_distribution)

plt.show()
