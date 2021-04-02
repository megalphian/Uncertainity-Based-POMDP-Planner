import numpy as np

from scipy.linalg import sqrtm

class KalmanEstimator:

    def __init__(self, A, C, M, environment):
        self.x_est = None
        self.cov_est = None

        self.belief_states = None
        self.W = None

        self.A = A
        self.C = C
        self.M = M
        self.environment = environment

    def get_estimates(self, no_iters, init_state, final_state, inputs, init_cov):
        cov = sqrtm(init_cov)
        x = np.asarray([[init_state[0]], [init_state[1]]])
        x_actual = x

        self.x_est = [x]
        self.cov_est = [cov]
        self.belief_states = [[x, cov]]
        self.W = [[0, 0]]

        for i in range(no_iters):

            C = self.C[i]
            A = self.A[i]
            M = self.M[i]

            measurement, N = self.environment.get_measurement(x_actual.flatten())

            A_cov = A @ cov
            tau = (A_cov) @ (np.transpose(A_cov)) + (M @ np.transpose(M))

            K = (tau @ np.transpose(C)) @ np.linalg.inv((C @ tau @ np.transpose(C)) + (N @ np.transpose(N)))

            w_term = sqrtm(K @ C @ tau)
            cov = sqrtm(tau - (K @ C @ tau))
            
            input_i = np.asarray([[inputs[i][0]], [inputs[i][1]]])
            x_actual = A @ x + input_i

            x = x_actual + (K @ (measurement - (C @ x)))

            self.belief_states.append([x_actual, cov])
            self.W.append([w_term, 0])

            self.x_est.append(x)
            self.cov_est.append(cov)

        self.x_est.append(np.asarray([[final_state[0]], [final_state[1]]]))

class POMDPController:

    def __init__(self, optimal_path):

        self.b_bar = optimal_path
        self.u_bar = self.compute_u_bar()

        self.Q_t = np.identity(2)
        self.R_t = np.identity(2)
        self.Q_l = 10 * len(self.b_bar) * np.identity(2)

    def compute_u_bar(self):

        u_bar = list()

        for i in range(len(self.b_bar) - 1):
            b_i = self.b_bar[i]
            b_j = self.b_bar[i+1]
            
            u_x = b_j[0] - b_i[0]
            u_y = b_j[1] - b_i[1]

            u_bar.append([u_x, u_y])
        
        return u_bar

    def calculate_linearized_belief_dynamics(self, beliefs, inputs):

        pass
        # Calculate Gt
        
        # belief 0 -> - h
        # belief 1 -> + h
        # (belief 1 - belief 0) / (x1 - x0), (belief 1 - belief 0) / (y1 - y0), (belief 1 - belief 0) / (sigma1[0] - sigma0[0])
        
        # Calculate Ft
        
        # (belief 1 - belief 0) / (u1 - u0), (belief 1 - belief 0) / (u1 - u0)

        # Calculate Gti
        # Calculate Fti

        # Calculate vectors
    
    def calculate_value_matrices(self, belief_dynamics):

        pass
        # Calculate Dt
        # Calculate Et
        # Calculate Ct

        # St+1 = 10 * l * identity(6)
        # Calculate St
    
    def calculate_trajectory_cost(self):

        pass
        # Calculate
    
    def get_new_path(self, x_est, path, old_u, start, end):
        new_u = list()
        new_path = list()
        for i in range(len(old_u) - 1):

            new_val_x = old_u[i][0] + 0.1 * (x_est[i+1][0] - path[i+1][0])[0]
            new_val_y = old_u[i][1] + 0.1 * (x_est[i+1][1] - path[i+1][1])[0]

            new_u.append([new_val_x, new_val_y])

        cur_point = [start[0], start[1]]
        new_path.append(cur_point)

        for i in range(len(new_u)):
            new_path_x = cur_point[0] + new_u[i][0]
            new_path_y = cur_point[1] + new_u[i][1]

            cur_point = [new_path_x, new_path_y]

            new_path.append(cur_point)
        
        last_u_x = end[0] - cur_point[0]
        last_u_y = end[1] - cur_point[1]

        new_u.append([last_u_x, last_u_y])
        new_path.append([end[0], end[1]])

        self.u_bar = new_u

        return [new_path, new_u]
