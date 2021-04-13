from ekf import KalmanEstimator
from pomdp_control import iLQGController

import numpy as np

class LocalUncertainityBasedPlanner:

    """
    Planner class to take an initial uncertainity-unaware path (a straight line path) 
    and create a new path that minimizes the uncertainity accrued along the path
    """

    def __init__(self, optimal_path, env, common_config, cost_function_config):

        start_hat = optimal_path.start_point.start_hat
        init_cov_mat = optimal_path.start_point.init_cov_mat

        # Initialize the EKF Estimator and iLQG controller to iterate along the POMDP belief space

        self.estimator = KalmanEstimator(env, common_config.time_step, optimal_path.get_control_length(), 0)
        self.controller = iLQGController(env, optimal_path.get_path_length(), cost_function_config, self.estimator)

        # Make initial belief estimates using the EKF based on the initial straight-line path
        
        self.current_u = np.array([np.array(xi).reshape((2,1)) for xi in optimal_path.inputs])
        self.estimator.make_EKF_Estimates(start_hat, self.current_u, init_cov_mat)

        self.initial_x_est = self.estimator.x_est
        self.current_path = self.estimator.belief_states

        self.goal = common_config.goal

    def optimize_plan_with_uncertainity(self):
        
        # Linearize the belief dynamics along the current optimal path

        belief_dynamics = self.controller.calculate_linearized_belief_dynamics(self.current_path, self.current_u)

        # Use the linearized belief dynamics and calculate the matrices required to calculate the value function and their gradients
        # We also note the initial trajectory cost to see how much the new path improves the belief certainity by

        valueData = self.controller.calculate_value_matrices(belief_dynamics, self.current_path, self.current_u)
        trajectory_cost = self.controller.calculate_trajectory_cost(self.current_path, self.current_u, belief_dynamics, valueData, self.goal)
        print('Original Trajectory Cost: ' + str(trajectory_cost))

        start_cost = trajectory_cost

        # Initialize a step size and iteration cap to ensure convergence

        original_step_size = 1
        step_size = original_step_size
        iteration_cap = 100

        while(True):

            # Using the gradients determined for the last uncertainity minimizing path, generate a new path and new control set
            # We start with a gradient descent step size of 1.
            # If the new path is more expensive, the step size is halved until the new path is less expensive.

            line_search_iterations = 0
            while(line_search_iterations < iteration_cap):
                new_path, new_u = self.controller.get_new_path(self.current_path, self.current_u, step_size, valueData)
                new_trajectory_cost = self.controller.calculate_trajectory_cost(new_path, new_u, belief_dynamics, valueData, self.goal)
                
                if(new_trajectory_cost < trajectory_cost):
                    step_size = original_step_size
                    break
                
                line_search_iterations += 1
                step_size = step_size/2

            # Break if the new trajectory is as expensive or more than the last optimal trajectory

            if(trajectory_cost <= new_trajectory_cost):
                break
            
            # Update the trajectory and its cost to reflect that we have found an optimal trajectory

            trajectory_cost = new_trajectory_cost
            self.current_u = new_u
            self.current_path = new_path

            print('Updated Path! Cost: ' + str(trajectory_cost))

            # Recalculate the belief states for the new trajectory

            start_belief = self.current_path[0]
            start_new = start_belief[0:2].flatten()

            cov = start_belief[2:]
            cov = cov.reshape((2,2))
            self.estimator.make_EKF_Estimates(start_new, self.current_u, cov)

            # Linearize the belief dynamics around the current trajectory and recalculate the value matrices.

            belief_dynamics = self.controller.calculate_linearized_belief_dynamics(self.current_path, self.current_u)
            valueData = self.controller.calculate_value_matrices(belief_dynamics, self.current_path, self.current_u)

        # Print the final trajectory's cost and the difference

        print('==================================================')
        print('Final Trajectory cost: ' + str(trajectory_cost))
        print('Cost Reduced by: ' + str(start_cost - trajectory_cost))