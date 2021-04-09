from environment import Environment
from optimalPlanner import StraightLinePlanner
from pomdp_controller import POMDPController

from descartes import PolygonPatch

from shapely.geometry import Polygon
from matplotlib import pyplot as plt

from stateNode import init_system_matrices
from ukf import KalmanEstimator

import numpy as np

rect_limits = [-5, 20]
resolution = 0.1
path_resolution = 0.5
time_step = 0.1

start_hat = (1, 12)

end = (12, 5)
cov_val = 1
init_covariance = cov_val * np.identity(2)
start_x = np.random.normal(start_hat[0], cov_val)
start_y = np.random.normal(start_hat[1], cov_val)

start = (start_x, start_y)

obstacles = []

env = Environment(rect_limits, resolution)
opt_planner = StraightLinePlanner(start, end, path_resolution, time_step)
path, inputs = opt_planner.generatePath()

controller = POMDPController(path, env)

A, C, M, _ = init_system_matrices(len(inputs), 0)

estimator = KalmanEstimator(A, C, M, env, time_step)

current_u = inputs.copy()

estimator.make_EKF_Estimates(start, current_u, init_covariance)
initial_x_est = estimator.x_est
current_path = estimator.belief_states

# TODO: Implement line search with the expected cost function
step_size = 0.001
trajectory_cost = np.inf
epsilon = 20

for i in range(10):
    controller.calculate_linearized_belief_dynamics(current_path, estimator.W1, estimator.W2, current_u, estimator, time_step)
    controller.calculate_value_matrices(current_path, current_u)

    new_path, new_u = controller.get_new_path(current_path, current_u, start, estimator, step_size)
    new_trajectory_cost = controller.calculate_trajectory_cost(new_path[-1])
    print(new_trajectory_cost)

    if(trajectory_cost - new_trajectory_cost < epsilon):
        break

    trajectory_cost = new_trajectory_cost
    current_u = new_u
    current_path = new_path

    start_belief = current_path[0]
    start_new =start_belief[0:2].reshape((2,))
    cov = start_belief[2:]
    cov = cov.reshape((2,2))
    estimator.make_EKF_Estimates(start_new, current_u, cov)

print('Final Trajectory cost: ' + str(trajectory_cost))

fig, ax = plt.subplots()

for coord in env.coords:
    ax.plot(coord[0], coord[1])

for obstacle in obstacles:
    patch = PolygonPatch(obstacle)
    ax.add_patch(patch)

x, y = zip(*env.sampled_points)
ax.scatter(x, y, c=env.uncertainity_distribution, cmap='winter_r')

ax.plot([x for (x, y) in path], [y for (x, y) in path], '.-', color='#083D77')
ax.plot([x for (x, y) in initial_x_est], [y for (x, y) in initial_x_est], '.-.', color='#D00000')
ax.plot([x for (x, y, _, _, _, _) in current_path], [y for (x, y, _, _, _, _) in current_path], '--o')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(x,y,env.uncertainity_distribution)

plt.show()
