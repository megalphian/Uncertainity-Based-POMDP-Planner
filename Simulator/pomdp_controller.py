import numpy as np

class KalmanEstimator:

    def __init__(self, A, C, M, environment):
        self.x_est = None
        self.cov_est = None

        self.A = A
        self.C = C
        self.M = M
        self.environment = environment

    def get_estimates(self, no_iters, init_state, inputs):
        cov = np.identity(2)
        x = np.asarray([[init_state[0]], [init_state[1]]])
        self.x_est = [x]
        self.cov_est = [cov]
        for i in range(no_iters):

            C = self.C[i]
            A = self.A[i]

            measurement, N = self.environment.get_measurement(x.flatten())

            sigma_cT = cov @ np.transpose(C)

            noise_term = (C @ sigma_cT) + N

            K = (A @ sigma_cT) * np.linalg.inv(noise_term)
            update = (A - (K @ C))

            measurement_i = np.asarray([[measurement]])
            input_i = np.asarray([[inputs[i][0]], [inputs[i][1]]])

            x = (update @ x) + (K @ measurement_i) + input_i

            cov = (update @ (cov @ np.linalg.inv(update))) + self.M[i] + (K @ N @ np.transpose(K))

            self.x_est.append(x)
            self.cov_est.append(cov)

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
