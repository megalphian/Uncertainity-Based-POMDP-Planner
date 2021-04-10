import numpy as np

class BeliefDynamicsData:

    def __init__(self):
        self.F = None
        self.Fi_1 = None
        self.Fi_2 = None

        self.G = None
        self.Gi_1 = None
        self.Gi_2 = None

        self.ei_1 = None
        self.ei_2 = None

class POMDPController:

    def __init__(self, optimal_path, environment):

        self.path_0 = optimal_path
        self.u_bar = None

        self.environment = environment

        self.Q_t = np.identity(6)
        self.R_t = np.identity(2)
        self.Q_l = 10 * len(self.path_0) * np.identity(6)

        self.trajectory_cost = np.inf

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

        h = 0.5

        linearized_dynamics = BeliefDynamicsData()

        linearized_dynamics.F = list()
        linearized_dynamics.Fi_1 = list()
        linearized_dynamics.Fi_2 = list()

        linearized_dynamics.G = list()
        linearized_dynamics.Gi_1 = list()
        linearized_dynamics.Gi_2 = list()

        linearized_dynamics.ei_1 = list()
        linearized_dynamics.ei_2 = list()

        for i in range(len(inputs)):

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

                x_1 = b_1[0:2].reshape((2,1))
                cov_1 = b_1[2:]
                cov_1 = cov_1.reshape((2,2))

                x_2 = b_2[0:2].reshape((2,1))
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

            Ft = np.hstack(Ft)
            Fit_1 = np.hstack(Fit_1)
            Fit_2 = np.hstack(Fit_2)
            
            linearized_dynamics.F.append(Ft)
            linearized_dynamics.Fi_1.append(Fit_1)
            linearized_dynamics.Fi_2.append(Fit_2)

            Gt = []
            Git_1 = []
            Git_2 = []

            for j in range(len(input_i)):

                input_1 = input_i.copy()
                input_2 = input_i.copy()

                input_1[j] += h
                input_2[j] -= h

                x = belief[0:2].reshape((2,1))
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

            Gt = np.hstack(Gt)
            linearized_dynamics.G.append(Gt)

            Git_1 = np.hstack(Git_1)
            Git_2 = np.hstack(Git_2)
            
            linearized_dynamics.Gi_1.append(Git_1)
            linearized_dynamics.Gi_2.append(Git_2)

            linearized_dynamics.ei_1.append(estimator.W1[i].reshape((6,1)))
            linearized_dynamics.ei_2.append(estimator.W2[i].reshape((6,1)))
        
        return linearized_dynamics
    
    def calculate_terminal_cost(self, terminal_belief):
        cost = np.transpose(terminal_belief) @ self.Q_l @ terminal_belief
        return cost[0][0]

    def calculate_stage_cost(self, belief, input_i):
        cov = belief[2:]
        trace = (np.transpose(cov) @ cov) # Ignoring Q_t as it is identity
        cost = trace + (np.transpose(input_i) @ self.R_t @ input_i)
        return cost[0][0]

    def calculate_value_matrices(self, belief_dynamics, beliefs, inputs):

        self.D = list()
        self.E = list()
        self.C = list()

        self.S = list()
        self.L = list()
        self.l = list()

        S_tplus1 = self.Q_l
        self.S.insert(0, S_tplus1)

        belief_l = beliefs[-1]
        s_tplus1_vec = []
        h = 0.5

        for j in range(len(belief_l)):
            b_1 = belief_l.copy()
            b_2 = belief_l.copy()

            b_1[j] += h
            b_2[j] -= h

            c1 = self.calculate_terminal_cost(b_1)
            c2 = self.calculate_terminal_cost(b_2)

            c_diff = (c1 - c2) / (2*h)
            s_tplus1_vec.append(c_diff)

        s_tplus1_vec = np.vstack(s_tplus1_vec)
        
        s_tplus1 = self.calculate_terminal_cost(belief_l)

        for t in reversed(range(len(inputs))):

            belief = beliefs[t]
            input_i = inputs[t]

            # Calculate Dt
            D_t = self.R_t + (np.transpose(belief_dynamics.G[t]) @ S_tplus1 @ belief_dynamics.G[t]) + (np.transpose(belief_dynamics.Gi_1[t]) @ S_tplus1 @ belief_dynamics.Gi_1[t]) + (np.transpose(belief_dynamics.Gi_2[t]) @ S_tplus1 @ belief_dynamics.Gi_2[t])
            self.D.insert(0, D_t)

            r_t = []

            for j in range(len(input_i)):

                input_1 = input_i.copy()
                input_2 = input_i.copy()

                input_1[j] += h
                input_2[j] -= h

                cost_1 = self.calculate_stage_cost(belief, input_1)
                cost_2 = self.calculate_stage_cost(belief, input_2)

                cost_diff = (cost_1 - cost_2) / (2*h)
                r_t.append(cost_diff)
            
            r_t = np.hstack(r_t).reshape((2,1))

            d_t = r_t + (np.transpose(belief_dynamics.G[t]) @ s_tplus1_vec) # Ignoring Gi as it is just 0

            q_t = []

            for j in range(len(belief)):

                belief_1 = belief.copy()
                belief_2 = belief.copy()

                belief_1[j] += h
                belief_2[j] -= h

                cost_1 = self.calculate_stage_cost(belief_1, input_i)
                cost_2 = self.calculate_stage_cost(belief_2, input_2)

                cost_diff = (cost_1 - cost_2) / (2*h)
                q_t.append(cost_diff)
            
            q_t = np.hstack(q_t).reshape((6,1))

            # Calculate Et
            E_t = (np.transpose(belief_dynamics.G[t]) @ S_tplus1 @ belief_dynamics.F[t]) + (np.transpose(belief_dynamics.Gi_1[t]) @ S_tplus1 @ belief_dynamics.Fi_1[t]) + (np.transpose(belief_dynamics.Gi_2[t]) @ S_tplus1 @ belief_dynamics.Fi_2[t])
            self.E.insert(0, E_t)

            L_t = - (np.linalg.inv(D_t) @ E_t)
            self.L.insert(0, L_t)

            l_t = - (np.linalg.inv(D_t) @ d_t)
            self.l.insert(0, l_t)

            mat_1 = (belief_dynamics.F[t] + (belief_dynamics.G[t] @ L_t))
            mat_2 = (belief_dynamics.Fi_1[t] + (belief_dynamics.Gi_1[t] @ L_t))
            mat_3 = (belief_dynamics.Fi_2[t] + (belief_dynamics.Gi_2[t] @ L_t))
            
            S_t = self.Q_t + (np.transpose(L_t) @ self.R_t @ L_t) + (np.transpose(mat_1) @ S_tplus1 @ mat_1) + (np.transpose(mat_2) @ S_tplus1 @ mat_2) + (np.transpose(mat_3) @ S_tplus1 @ mat_3)

            self.S.insert(0, S_t)
            S_tplus1 = S_t

            ei_1 = belief_dynamics.ei_1[t]
            ei_2 = belief_dynamics.ei_2[t]

            s_tplus1_vec = q_t + (np.transpose(L_t) @ r_t) + (np.transpose(mat_1) @ s_tplus1_vec) + (np.transpose(mat_1) @ S_tplus1 @ ei_1) + (np.transpose(mat_1) @ S_tplus1 @ ei_2)

            cost = self.calculate_stage_cost(belief, input_i)
            stage_addition = np.real((cost + ((1/2) * ((np.transpose(ei_1) @ S_tplus1 @ ei_1) + (np.transpose(ei_2) @ S_tplus1 @ ei_2)))))[0][0]
            s_tplus1 += stage_addition

            # # Calculate Ct
            # C_t = self.Q_t + (np.transpose(belief_dynamics.F[t]) @ S_tplus1 @ belief_dynamics.F[t]) 
            # # + (np.transpose(belief_dynamics.Fi_1[t]) @ S_tplus1 @ belief_dynamics.Fi_1[t]) + (np.transpose(belief_dynamics.Fi_2[t]) @ S_tplus1 @ belief_dynamics.Fi_2[t])
            # self.C.insert(0, C_t)

            # c_t = q_t + (np.transpose(belief_dynamics.F[t]) @ s_tplus1) + (np.transpose(belief_dynamics.Fi_1[t]) @ S_tplus1 @ belief_dynamics.ei_1[t]) + (np.transpose(belief_dynamics.Fi_2[t]) @ S_tplus1 @ belief_dynamics.ei_2[t])

        return s_tplus1

    def get_new_path(self, beliefs, old_u, estimator, step_size):
        new_u = list()
        
        new_path = list()
        new_path.append(beliefs[0])

        cur_belief = beliefs[0]

        for i in range(len(old_u)): 
            u_bar = old_u[i]
            b_bar = beliefs[i]

            C = estimator.C[i]
            A = estimator.A[i]
            M = estimator.M[i]

            x = cur_belief[0:2].reshape((2,1))
            cov = cur_belief[2:]
            cov = cov.reshape((2,2))

            measurement, N = self.environment.get_measurement(x.flatten())

            new_u_val = u_bar + (self.L[i] @ (cur_belief - b_bar)) + (step_size * self.l[i])
            new_u.append(new_u_val)

            new_b, _, _, _, _, _ = estimator.make_estimate(A, C, M, N, cov, x, measurement, new_u_val)
            new_path.append(new_b)
            cur_belief = new_b

        new_path = np.real(new_path)
        new_u = np.real(new_u)
        return [new_path, new_u]
