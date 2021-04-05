from environment import Environment
from optimalPlanner import StraightLinePlanner
from pomdp_controller import POMDPController

from descartes import PolygonPatch

from shapely.geometry import Polygon
from matplotlib import pyplot as plt

from stateNode import init_system_matrices
from pomdp_controller import KalmanEstimator

import numpy as np

rect_limits = [-5, 20]
resolution = 0.1
path_resolution = 0.5
time_step = 0.1

start_hat = (1, 12)

end = (12, 0)
init_covariance = 0.25 * np.identity(2)
start_x = np.random.normal(start_hat[0], 0.25)
start_y = np.random.normal(start_hat[1], 0.25)

start = (start_x, start_y)

obstacles = []

env = Environment(rect_limits, resolution)
opt_planner = StraightLinePlanner(start, end, path_resolution, time_step)
path, inputs = opt_planner.generatePath()
# path.reverse()

controller = POMDPController(path, env)

A, C, M, _ = init_system_matrices(len(inputs), 0)

estimator = KalmanEstimator(A, C, M, env, time_step)

new_u = inputs.copy()

estimator.make_EKF_Estimates(start, new_u, init_covariance)
new_path = estimator.belief_states

for i in range(8):
    controller.calculate_linearized_belief_dynamics(new_path, estimator.W1, estimator.W2, new_u, estimator, time_step)
    controller.calculate_value_matrices(new_path, new_u)
    new_path, new_u = controller.get_new_path(new_path, new_u, start, estimator)

fig, ax = plt.subplots()

for coord in env.coords:
    ax.plot(coord[0], coord[1])

for obstacle in obstacles:
    patch = PolygonPatch(obstacle)
    ax.add_patch(patch)

x, y = zip(*env.sampled_points)
ax.scatter(x, y, c=env.uncertainity_distribution, cmap='winter_r')

ax.plot([x for (x, y) in path], [y for (x, y) in path], '.-', color='#083D77')
ax.plot([x for (x, y) in estimator.x_est], [y for (x, y) in estimator.x_est], '.-.', color='#D00000')
ax.plot([x for (x, y, _, _, _, _) in new_path], [y for (x, y, _, _, _, _) in new_path], '--o')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(x,y,env.uncertainity_distribution)

plt.show()
