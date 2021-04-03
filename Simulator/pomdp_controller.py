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

    def make_EKF_Estimates(self, init_state, final_state, inputs, init_cov):
        cov = sqrtm(init_cov)
        x = np.asarray([[init_state[0]], [init_state[1]]])
        x_actual = x

        self.x_est = [x]
        self.cov_est = [cov]
        self.belief_states = [np.concatenate((x.flatten(), cov.flatten()))]
        self.W = [[0, 0]]

        for i in range(len(inputs)):

            C = self.C[i]
            A = self.A[i]
            M = self.M[i]

            measurement, N = self.environment.get_measurement(x_actual.flatten())
            input_i = np.asarray([[inputs[i][0]], [inputs[i][1]]])

            belief, w_term, x, cov, x_actual = self.make_estimate(A, C, M, N, cov, x, measurement, input_i)

            self.belief_states.append(belief)
            self.W.append([w_term, 0])

            self.x_est.append(x)
            self.cov_est.append(cov)

        self.x_est.append(np.asarray([[final_state[0]], [final_state[1]]]))

    def make_estimate(self, A, C, M, N, cov, x, measurement, input_i):

        A_cov = A @ cov
        tau = (A_cov) @ (np.transpose(A_cov)) + (M @ np.transpose(M))

        K = (tau @ np.transpose(C)) @ np.linalg.inv((C @ tau @ np.transpose(C)) + (N @ np.transpose(N)))

        w_term = sqrtm(K @ C @ tau)
        cov = sqrtm(tau - (K @ C @ tau))
        
        x_actual = A @ x + input_i

        belief = np.concatenate((x_actual.flatten(), cov.flatten()))

        x = x_actual + (K @ (measurement - (C @ x)))

        return (belief, w_term, x, cov, x_actual)

class POMDPController:

    def __init__(self, optimal_path, environment):

        self.path_0 = optimal_path
        self.u_bar = self.compute_u_bar()

        self.environment = environment

        self.Q_t = np.identity(2)
        self.R_t = np.identity(2)
        self.Q_l = 10 * len(self.path_0) * np.identity(2)

    def compute_u_bar(self):

        u_bar = list()

        for i in range(len(self.path_0) - 1):
            b_i = self.path_0[i]
            b_j = self.path_0[i+1]
            
            u_x = b_j[0] - b_i[0]
            u_y = b_j[1] - b_i[1]

            u_bar.append([u_x, u_y])
        
        return u_bar

    def calculate_linearized_belief_dynamics(self, beliefs, inputs, estimator):

        h = 0.1

        F = list()
        Fi_1 = list()
        Fi_2 = list()

        G = list()
        Gi_1 = list()
        Gi_2 = list()

        for i in range(len(beliefs) - 1):
        # for i in range(1):

            belief = beliefs[i]
            input_i = inputs[i]

            C = estimator.C[i]
            A = estimator.A[i]
            M = estimator.M[i]

            Ft = []
            Fit_1 = []
            Fit_2 = []

            for j in range(len(belief)):
                b_1 = belief.copy()
                b_2 = belief.copy()

                b_1[j] += h
                b_2[j] -= h

                x_1 = b_1[0:2]
                cov_1 = b_1[2:]
                cov_1 = cov_1.reshape((2,2))

                x_2 = b_2[0:2]
                cov_2 = b_2[2:]
                cov_2 = cov_2.reshape((2,2))

                measurement_1, N_1 = self.environment.get_measurement(x_1.flatten())
                measurement_2, N_2 = self.environment.get_measurement(x_2.flatten())

                g_1, w_1, _, _, _ = estimator.make_estimate(A, C, M, N_1, cov_1, x_1, measurement_1, input_i)
                g_2, w_2, _, _, _ = estimator.make_estimate(A, C, M, N_2, cov_2, x_2, measurement_2, input_i)

                w1_diff = (w_1[:,0] - w_2[:,0]) / (2*h)
                w2_diff = (w_1[:,1] - w_2[:,1]) / (2*h)

                zeros = np.zeros((1,4))

                w1_diff = np.concatenate((w1_diff.flatten(), zeros.flatten()))
                w2_diff = np.concatenate((w2_diff.flatten(), zeros.flatten()))

                Fit_1.append(w1_diff)
                Fit_2.append(w2_diff)

                g_diff = (g_1 - g_2) / (2*h)
                Ft.append(g_diff)

            Ft = np.transpose(np.vstack(Ft))
            Fit_1 = np.transpose(np.vstack(Fit_1))
            Fit_2 = np.transpose(np.vstack(Fit_2))
            
            F.append(Ft)
            Fi_1.append(Fit_1)
            Fi_2.append(Fit_2)

            Gt = []
            Git_1 = []
            Git_2 = []

            for j in range(len(input_i)):

                input_1 = input_i.copy()
                input_2 = input_i.copy()

                input_1[j] += h
                input_2[j] -= h

                x = belief[0:2]
                cov = belief[2:]
                cov = cov.reshape((2,2))

                measurement, N = self.environment.get_measurement(x.flatten())

                g_1, w_1, _, _, _ = estimator.make_estimate(A, C, M, N, cov, x, measurement, input_1)
                g_2, w_2, _, _, _= estimator.make_estimate(A, C, M, N, cov, x, measurement, input_2)

                w1_diff = (w_1[:,0] - w_2[:,0]) / (2*h)
                w2_diff = (w_1[:,1] - w_2[:,1]) / (2*h)

                zeros = np.zeros((1,4))

                w1_diff = np.concatenate((w1_diff.flatten(), zeros.flatten()))
                w2_diff = np.concatenate((w2_diff.flatten(), zeros.flatten()))

                g_diff = (g_1 - g_2) / (2*h)

                Gt.append(g_diff)

            Gt = np.transpose(np.vstack(Gt))
            G.append(Gt)

            Git_1 = np.transpose(np.vstack(Git_1))
            Git_2 = np.transpose(np.vstack(Git_2))
            
            Gi_1.append(Git_1)
            Gi_2.append(Git_2)

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
