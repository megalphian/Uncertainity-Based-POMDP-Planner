from environment import Environment
from optimalPlanner import RRTPlanner
from pomdp_controller import POMDPController

from descartes import PolygonPatch

from shapely.geometry import Polygon
from matplotlib import pyplot as plt

from stateNode import init_system_matrices
from pomdp_controller import KalmanEstimator

import numpy as np

rect_limits = [-2, 15]
resolution = 0.1

end = (1.5, 2.5)

start_hat = (7.5, 12.5)
init_covariance = 0.25 * np.identity(2)
start_x = np.random.normal(start_hat[0], 0.25)
start_y = np.random.normal(start_hat[1], 0.25)

start = (start_x, start_y)

# obstacles = [Polygon(((6,-2), (6,6), (5,6), (5,-2))), Polygon(((6,11), (6,15), (5,15), (5,11)))]
obstacles = []

env = Environment(rect_limits, resolution)
rrt = RRTPlanner(start, end, obstacles, rect_limits)
path = rrt.planning()
path.reverse()

controller = POMDPController(path, env)

inputs = controller.u_bar

A, C, M, _ = init_system_matrices(len(inputs), 0)

estimator = KalmanEstimator(A, C, M, env)

new_path = path.copy()
new_u = inputs.copy()

for i in range(7):
    estimator.make_EKF_Estimates(start, end, new_u, init_covariance)
    controller.calculate_linearized_belief_dynamics(estimator.belief_states, estimator.W1, estimator.W2, new_u, estimator)
    controller.calculate_value_matrices(estimator.belief_states, new_u)
    new_path, new_u = controller.get_new_path(estimator.belief_states, new_u, start, end, estimator)

fig, ax = plt.subplots()

for coord in env.coords:
    ax.plot(coord[0], coord[1])

for obstacle in obstacles:
    patch = PolygonPatch(obstacle)
    ax.add_patch(patch)

x, y = zip(*env.sampled_points)
ax.scatter(x, y, c=env.uncertainity_distribution, cmap='winter_r')

ax.plot([x for (x, y) in path], [y for (x, y) in path], '.-', color='#083D77')
# ax.plot([x for (x, y) in estimator.x_est], [y for (x, y) in estimator.x_est], '.-.', color='#D00000')
ax.plot([x for (x, y, _, _, _, _) in new_path], [y for (x, y, _, _, _, _) in new_path], '--o')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(x,y,env.uncertainity_distribution)

plt.show()
