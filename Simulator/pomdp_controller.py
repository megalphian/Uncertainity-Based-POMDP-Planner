import numpy as np

from scipy.linalg import sqrtm

class BeliefDynamicsData:

    def __init__(self):
        self.F = None
        self.Fi_1 = None
        self.Fi_2 = None

        self.G = None
        self.Gi_1 = None
        self.Gi_2 = None

        self.beliefs = None
        self.inputs = None
        self.W = None


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

        self.Q_t = np.identity(6)
        self.R_t = np.identity(2)
        self.Q_l = 10 * len(self.path_0) * np.identity(6)

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

        self.steps = len(beliefs)

        self.F = list()
        self.Fi_1 = list()
        self.Fi_2 = list()

        self.G = list()
        self.Gi_1 = list()
        self.Gi_2 = list()

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
            
            self.F.append(Ft)
            self.Fi_1.append(Fit_1)
            self.Fi_2.append(Fit_2)

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

                Git_1.append(w1_diff)
                Git_2.append(w2_diff)

                g_diff = (g_1 - g_2) / (2*h)
                Gt.append(g_diff)

            Gt = np.transpose(np.vstack(Gt))
            self.G.append(Gt)

            Git_1 = np.transpose(np.vstack(Git_1))
            Git_2 = np.transpose(np.vstack(Git_2))
            
            self.Gi_1.append(Git_1)
            self.Gi_2.append(Git_2)

            # Calculate vectors
    
    def calculate_value_matrices(self):

        self.D = list()
        self.E = list()
        self.C = list()

        self.S = list()
        self.L = list()

        S_tplus1 = self.Q_l
        self.S.insert(0, S_tplus1)

        for i in range(len(self.G) - 1, 0, -1):

            # Calculate Dt
            D_t = self.R_t + (np.transpose(self.G[i]) @ S_tplus1 @ self.G[i]) + (np.transpose(self.Gi_1[i]) @ S_tplus1 @ self.Gi_1[i]) + (np.transpose(self.Gi_2[i]) @ S_tplus1 @ self.Gi_2[i])
            self.D.insert(0, D_t)

            # TODO: Calculate d_t vector

            # Calculate Et
            E_t = (np.transpose(self.G[i]) @ S_tplus1 @ self.F[i]) + (np.transpose(self.Gi_1[i]) @ S_tplus1 @ self.Fi_1[i]) + (np.transpose(self.Gi_2[i]) @ S_tplus1 @ self.Fi_2[i])
            self.E.insert(0, E_t)

            # Calculate Ct
            C_t = self.Q_t + (np.transpose(self.F[i]) @ S_tplus1 @ self.F[i]) + (np.transpose(self.Fi_1[i]) @ S_tplus1 @ self.Fi_1[i]) + (np.transpose(self.Fi_2[i]) @ S_tplus1 @ self.Fi_2[i])
            self.C.insert(0, C_t)

            S_t = C_t - (np.transpose(E_t) @ np.linalg.inv(D_t) @ E_t)
            self.S.insert(0, S_t)

            L_t = - np.linalg.inv(D_t) @ E_t
            self.L.insert(0, L_t)
    
    def calculate_trajectory_cost(self):

        pass
        # Calculate
    
    def get_new_path(self, beliefs, old_u, start, end):
        new_u = list()
        new_path = list()
        for i in range(len(old_u) - 1):
            u_bar = np.array(old_u[i])
            x_bar = np.array(path[i+1])
            print(u_bar.shape)
            print(self.L[i].shape)

            new_u = u_bar + (self.L[i] @ (beliefs[i+1] - x_bar))

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
