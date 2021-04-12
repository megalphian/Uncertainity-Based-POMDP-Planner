from ekf import KalmanEstimator
from pomdp_controller import POMDPController

import numpy as np

class LocalUncertainityBasedPlanner:

    def __init__(self, optimal_path, env, config):

        start_hat = optimal_path.start_point.start_hat
        init_cov_mat = optimal_path.start_point.init_cov_mat

        self.controller = POMDPController(env, optimal_path.get_path_length(), config.cost_function_config)
        self.estimator = KalmanEstimator(env, config.common_config.time_step, optimal_path.get_control_length(), 0)
        
        self.current_u = np.array([np.array(xi).reshape((2,1)) for xi in optimal_path.inputs])
        self.estimator.make_EKF_Estimates(start_hat, self.current_u, init_cov_mat)

        self.initial_x_est = self.estimator.x_est
        self.current_path = self.estimator.belief_states

        self.goal = config.common_config.goal

    def optimize_plan_with_uncertainity(self):

        belief_dynamics = self.controller.calculate_linearized_belief_dynamics(self.current_path, self.current_u, self.estimator)

        self.controller.calculate_value_matrices(belief_dynamics, self.current_path, self.current_u)
        trajectory_cost = self.controller.calculate_trajectory_cost(self.current_path, self.current_u, belief_dynamics, self.goal)
        print('Original Trajectory Cost: ' + str(trajectory_cost))

        start_cost = trajectory_cost

        original_step_size = 1
        step_size = original_step_size
        iteration_cap = 100

        while(True):
            
            line_search_iterations = 0
            while(line_search_iterations < iteration_cap):
                new_path, new_u = self.controller.get_new_path(self.current_path, self.current_u, self.estimator, step_size)
                new_trajectory_cost = self.controller.calculate_trajectory_cost(new_path, new_u, belief_dynamics, self.goal)
                
                if(new_trajectory_cost <= trajectory_cost):
                    step_size = original_step_size
                    break
                
                line_search_iterations += 1
                step_size = step_size/2
                # print('New path is more expensive. Keep Looking...')

            if(trajectory_cost <= new_trajectory_cost):
                break

            trajectory_cost = new_trajectory_cost
            self.current_u = new_u
            self.current_path = new_path

            print('Updated Path! Cost: ' + str(trajectory_cost))

            start_belief = self.current_path[0]
            start_new = start_belief[0:2].flatten()

            cov = start_belief[2:]
            cov = cov.reshape((2,2))
            self.estimator.make_EKF_Estimates(start_new, self.current_u, cov)

            belief_dynamics = self.controller.calculate_linearized_belief_dynamics(self.current_path, self.current_u, self.estimator)
            self.controller.calculate_value_matrices(belief_dynamics, self.current_path, self.current_u)

        print('==================================================')
        print('Final Trajectory cost: ' + str(trajectory_cost))
        print('Cost Reduced by: ' + str(start_cost - trajectory_cost))