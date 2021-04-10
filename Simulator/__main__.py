from environment import Environment
from optimalPlanner import StraightLinePlanner
from pomdp_controller import POMDPController

from descartes import PolygonPatch

from shapely.geometry import Polygon
from matplotlib import pyplot as plt

from stateNode import init_system_matrices
from ukf import KalmanEstimator

import numpy as np

rect_limits = [-10, 15]
resolution = 0.1
path_resolution = 0.4
time_step = 0.1

start_hat = (10, 10)

end = (2, -2)
cov_val = 0.25
init_covariance = cov_val * np.identity(2)
start_x = np.random.normal(start_hat[0], cov_val)
start_y = np.random.normal(start_hat[1], cov_val)

start = np.array((start_x, start_y)).reshape((2,1))

obstacles = []

env = Environment(rect_limits, resolution)
opt_planner = StraightLinePlanner(start_hat, end, path_resolution, time_step)
path, inputs = opt_planner.generatePath()

controller = POMDPController(path, env)

A, C, M, _ = init_system_matrices(len(inputs), 0)

estimator = KalmanEstimator(A, C, M, env, time_step)

current_u = y=np.array([np.array(xi).reshape((2,1)) for xi in inputs])

estimator.make_EKF_Estimates(start_hat, current_u, init_covariance)
initial_x_est = estimator.x_est
current_path = estimator.belief_states

original_step_size = 1

step_size = original_step_size
trajectory_cost = np.inf
epsilon = 50
iteration_cap = 20

for i in range(10):
    belief_dynamics = controller.calculate_linearized_belief_dynamics(current_path, current_u, estimator)
    expected_cost = controller.calculate_value_matrices(belief_dynamics, current_path, current_u)
    
    line_search_iterations = 0
    while(line_search_iterations < iteration_cap):
        new_path, new_u = controller.get_new_path(current_path, current_u, estimator, step_size)
        new_belief_dynamics = controller.calculate_linearized_belief_dynamics(new_path, new_u, estimator)
        new_trajectory_cost = controller.calculate_value_matrices(new_belief_dynamics, new_path, new_u)
        # new_trajectory_cost = controller.calculate_trajectory_cost(new_path, new_u, belief_dynamics)
        
        print(new_trajectory_cost)
        if(new_trajectory_cost <= expected_cost or abs(trajectory_cost - new_trajectory_cost) < epsilon):
            step_size = original_step_size
            break
        
        line_search_iterations += 1
        step_size = step_size/2
        print('Keep looking')

    if(trajectory_cost - new_trajectory_cost < epsilon):
        break

    trajectory_cost = new_trajectory_cost
    current_u = new_u
    current_path = new_path

    # start_belief = current_path[0]
    # start_new =start_belief[0:2].reshape((2,))
    # cov = start_belief[2:]
    # cov = cov.reshape((2,2))
    # estimator.make_EKF_Estimates(start_new, current_u, cov)

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
