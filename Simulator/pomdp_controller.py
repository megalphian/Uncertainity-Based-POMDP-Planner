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

    def __init__(self, A, C, M, environment, time_step):
        self.x_est = None
        self.cov_est = None

        self.belief_states = None
        self.W1 = None
        self.W2 = None

        self.A = A
        self.C = C
        self.M = M
        self.environment = environment
        self.time_step = time_step

    def make_EKF_Estimates(self, init_state, inputs, init_cov):
        cov = sqrtm(init_cov)
        x = np.asarray([[init_state[0]], [init_state[1]]])
        x_actual = x

        self.x_est = [x]
        self.cov_est = [cov]
        self.belief_states = [np.concatenate((x.flatten(), cov.flatten()))]
        self.W1 = [np.zeros((1,6))]
        self.W2 = [np.zeros((1,6))]

        for i in range(len(inputs)):

            C = self.C[i]
            A = self.A[i]
            M = self.M[i]

            measurement, N = self.environment.get_measurement(x_actual.flatten())
            input_i = np.asarray([[inputs[i][0]], [inputs[i][1]]])

            belief, w1, w2, x, cov, x_actual = self.make_estimate(A, C, M, N, cov, x, measurement, input_i)

            self.belief_states.append(belief)

            self.W1.append(w1)
            self.W2.append(w2)

            self.x_est.append(x)
            self.cov_est.append(cov)

    def make_estimate(self, A, C, M, N, cov, x, measurement, input_i):

        A_cov = A @ cov
        tau = (A_cov) @ (np.transpose(A_cov)) + (M @ np.transpose(M))

        K = (tau @ np.transpose(C)) @ np.linalg.inv((C @ tau @ np.transpose(C)) + (N @ np.transpose(N)))

        w_term = sqrtm(K @ C @ tau)
        cov = sqrtm(tau - (K @ C @ tau))
        
        input_steps = [x * self.time_step for x in input_i]
        x_actual = A @ x + input_steps

        belief = np.concatenate((x_actual.flatten(), cov.flatten()))

        x = x_actual + (K @ (measurement - (C @ x)))

        zeros = np.zeros((1,4))
        w1 = np.concatenate((w_term[:,0].flatten(), zeros.flatten()))

        w2 = np.concatenate((w_term[:,1].flatten(), zeros.flatten()))

        return (belief, w1, w2, x, cov, x_actual)

class POMDPController:

    def __init__(self, optimal_path, environment):

        self.path_0 = optimal_path
        self.u_bar = None

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

    def calculate_linearized_belief_dynamics(self, beliefs, W1, W2, inputs, estimator):

        h = 0.1

        self.steps = len(beliefs)

        self.F = list()
        self.Fi_1 = list()
        self.Fi_2 = list()

        self.G = list()
        self.Gi_1 = list()
        self.Gi_2 = list()

        self.ei_1 = list()
        self.ei_2 = list()

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

                x_1 = b_1[0:2].reshape((2,))
                cov_1 = b_1[2:]
                cov_1 = cov_1.reshape((2,2))

                x_2 = b_2[0:2].reshape((2,))
                cov_2 = b_2[2:]
                cov_2 = cov_2.reshape((2,2))

                measurement_1, N_1 = self.environment.get_measurement(x_1.flatten())
                measurement_2, N_2 = self.environment.get_measurement(x_2.flatten())

                g_1, w1_1, w1_2, _, _, _ = estimator.make_estimate(A, C, M, N_1, cov_1, x_1, measurement_1, input_i)
                g_2, w2_1, w2_2, _, _, _ = estimator.make_estimate(A, C, M, N_2, cov_2, x_2, measurement_2, input_i)

                g_diff = (g_1 - g_2) / (2*h)
                w1_diff = (w1_1 - w2_1) / (2*h)
                w2_diff = (w1_2 - w2_2) / (2*h)

                Fit_1.append(w1_diff)
                Fit_2.append(w2_diff)
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

                x = belief[0:2].reshape((2,))
                cov = belief[2:]
                cov = cov.reshape((2,2))

                measurement, N = self.environment.get_measurement(x.flatten())

                g_1, w1_1, w1_2, _, _, _ = estimator.make_estimate(A, C, M, N, cov, x, measurement, input_1)
                g_2, w2_1, w2_2, _, _, _= estimator.make_estimate(A, C, M, N, cov, x, measurement, input_2)

                g_diff = (g_1 - g_2) / (2*h)
                w1_diff = (w1_1 - w2_1) / (2*h)
                w2_diff = (w1_2 - w2_2) / (2*h)

                Git_1.append(w1_diff)
                Git_2.append(w2_diff)
                Gt.append(g_diff)

            Gt = np.transpose(np.vstack(Gt))
            self.G.append(Gt)

            Git_1 = np.transpose(np.vstack(Git_1))
            Git_2 = np.transpose(np.vstack(Git_2))
            
            self.Gi_1.append(Git_1)
            self.Gi_2.append(Git_2)

            self.ei_1.append(W1[i])
            self.ei_2.append(W2[i])
    
    def calculate_terminal_cost(self, terminal_belief):
        cost = np.transpose(terminal_belief) @ self.Q_l @ terminal_belief
        return cost

    def calculate_stage_cost(self, belief):
        cost = np.transpose(belief) @ self.Q_l @ belief
        return cost

    def calculate_value_matrices(self, beliefs, inputs):

        self.D = list()
        self.E = list()
        self.C = list()

        self.S = list()
        self.L = list()
        self.l = list()

        S_tplus1 = self.Q_l
        self.S.insert(0, S_tplus1)

        belief_l = beliefs[-1]
        s_tplus1 = []
        h = 0.1

        for j in range(len(belief_l)):
            b_1 = belief_l.copy()
            b_2 = belief_l.copy()

            b_1[j] += h
            b_2[j] -= h

            c1 = self.calculate_terminal_cost(b_1)
            c2 = self.calculate_terminal_cost(b_2)

            c_diff = (c1 - c2) / (2*h)
            s_tplus1.append(c_diff)

        s_tplus1 = np.vstack(s_tplus1)
        s_tplus1.reshape((6,))

        for t in range(len(self.G) - 1, 0, -1):
        # for t in range(1):

            # Calculate Dt
            D_t = self.R_t + (np.transpose(self.G[t]) @ S_tplus1 @ self.G[t]) + (np.transpose(self.Gi_1[t]) @ S_tplus1 @ self.Gi_1[t]) + (np.transpose(self.Gi_2[t]) @ S_tplus1 @ self.Gi_2[t])
            self.D.insert(0, D_t)

            input_i = np.asarray([[inputs[t][0]], [inputs[t][1]]])
            d_t = (self.R_t @ input_i) + (np.transpose(self.G[t]) @ s_tplus1) # Ignoring Gi as it is just 0
            d_t = d_t.reshape(2,)

            # Calculate Et
            E_t = (np.transpose(self.G[t]) @ S_tplus1 @ self.F[t]) + (np.transpose(self.Gi_1[t]) @ S_tplus1 @ self.Fi_1[t]) + (np.transpose(self.Gi_2[t]) @ S_tplus1 @ self.Fi_2[t])
            self.E.insert(0, E_t)

            # Calculate Ct
            C_t = self.Q_t + (np.transpose(self.F[t]) @ S_tplus1 @ self.F[t]) + (np.transpose(self.Fi_1[t]) @ S_tplus1 @ self.Fi_1[t]) + (np.transpose(self.Fi_2[t]) @ S_tplus1 @ self.Fi_2[t])
            self.C.insert(0, C_t)

            c_t = (self.Q_t @ beliefs[t]) + (np.transpose(self.F[t]) @ s_tplus1).reshape((6,)) + (np.transpose(self.Fi_1[t]) @ S_tplus1 @ self.ei_1[t]) + (np.transpose(self.Fi_2[t]) @ S_tplus1 @ self.ei_2[t])

            S_t = C_t - (np.transpose(E_t) @ np.linalg.inv(D_t) @ E_t)
            self.S.insert(0, S_t)

            S_tplus1 = S_t

            L_t = - np.linalg.inv(D_t) @ E_t
            self.L.insert(0, L_t)

            l_t = - np.linalg.inv(D_t) @ d_t
            self.l.insert(0, l_t)
    
    def calculate_trajectory_cost(self):

        pass
        # Calculate
    
    def get_new_path(self, beliefs, old_u, start, estimator):
        new_u = list()
        
        new_path = list()
        new_path.append(beliefs[0])

        cur_belief = beliefs[0]

        for i in range(len(old_u) - 1): 
            u_bar = np.array(old_u[i])
            b_bar = np.array(beliefs[i])

            C = estimator.C[i]
            A = estimator.A[i]
            M = estimator.M[i]

            x = cur_belief[0:2].reshape((2,))
            cov = cur_belief[2:]
            cov = cov.reshape((2,2))

            measurement, N = self.environment.get_measurement(x.flatten())

            new_u_val = u_bar + (self.L[i] @ (cur_belief - b_bar)) + (0.001 * self.l[i]) # Using constant step size, but can use line search here
            new_u.append(new_u_val)

            new_b, _, _, _, _, _ = estimator.make_estimate(A, C, M, N, cov, x, measurement, new_u_val)
            new_path.append(new_b)
            cur_belief = new_b

        return [new_path, new_u]
