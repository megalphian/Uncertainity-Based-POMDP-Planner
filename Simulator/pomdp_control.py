import numpy as np

class BeliefDynamicsData:

    """
    Data class to store the Belief Dynamics Data linearized about the optimal trajectory
    """

    def __init__(self):

        self.F = list()
        self.Fi_1 = list()
        self.Fi_2 = list()

        self.G = list()
        self.Gi_1 = list()
        self.Gi_2 = list()

        self.ei_1 = list()
        self.ei_2 = list()

class QuadratizedValueData:

    """
    Data class to store the quadratized value functions required to compute cost and calculate a new path
    """

    def __init__(self):

        self.L = list()
        self.l = list()

class iLQGController:

    """
    The Controller for the iLQG problem of minimizing the cost of accruing uncertainity along an inital optimal trajectory.
    This class computes the linearized belief dynamics for a given trajectory, obtains the matrices to compute the quadratized 
    value function using the dynamics and creates a new path using the gradients computed for the value function.
    """

    def __init__(self, environment, steps, cost_config, estimator):

        self.environment = environment
        self.estimator = estimator

        terminal_cost_multiplier = cost_config.terminal_cost_multiplier
        stage_cost_multiplier = cost_config.stage_cost_multiplier

        # Initialize the matrices to compute the stage and terminal costs for the LQG problem.

        self.Q_t = stage_cost_multiplier * np.identity(6)
        self.R_t = stage_cost_multiplier * np.identity(2)
        self.Q_l = terminal_cost_multiplier * steps * np.identity(6)

        # Symmetric step size for numerical differentiation
        self.h = 0.1

    def calculate_linearized_belief_dynamics(self, beliefs, inputs):

        # Linearize belief dynamics by calculating the gradients of the system dynamics wrt to changes in 
        # the belief and input at each point of the trajectory

        linearized_dynamics = BeliefDynamicsData()

        for i in range(len(inputs)):

            belief = beliefs[i]
            input_i = inputs[i]

            Ft, Fit_1, Fit_2 = self._calculate_belief_gradients(belief, input_i, i)
            Gt, Git_1, Git_2 = self._calculate_input_gradients(belief, input_i, i)

            linearized_dynamics.F.append(Ft)
            linearized_dynamics.Fi_1.append(Fit_1)
            linearized_dynamics.Fi_2.append(Fit_2)
            
            linearized_dynamics.G.append(Gt)
            linearized_dynamics.Gi_1.append(Git_1)
            linearized_dynamics.Gi_2.append(Git_2)

            linearized_dynamics.ei_1.append(self.estimator.W1[i].reshape((6,1)))
            linearized_dynamics.ei_2.append(self.estimator.W2[i].reshape((6,1)))
        
        return linearized_dynamics

    def calculate_value_matrices(self, belief_dynamics, beliefs, inputs):

        # Calculate the quadratized value function matrix and gradient vector for the terminal belief 
        # This uses the linearized belief dynamics along the optimal trajectory

        valueData = QuadratizedValueData()

        S_tplus1 = self.Q_l

        belief_l = beliefs[-1]
        s_tplus1_vec = self._calculate_terminal_cost_gradient(belief_l)

        for t in reversed(range(len(inputs))):

            belief = beliefs[t]
            input_i = inputs[t]

            # Calculate Dt
            D_t = self.R_t + (np.transpose(belief_dynamics.G[t]) @ S_tplus1 @ belief_dynamics.G[t]) + (np.transpose(belief_dynamics.Gi_1[t]) @ S_tplus1 @ belief_dynamics.Gi_1[t]) + (np.transpose(belief_dynamics.Gi_2[t]) @ S_tplus1 @ belief_dynamics.Gi_2[t])

            r_t = self._calculate_gradient_input_stage_cost(belief, input_i)

            # Calculate d_t (gradient of value wrt input)

            d_t = r_t + (np.transpose(belief_dynamics.G[t]) @ s_tplus1_vec) # Ignoring Gi as it is just 0

            # Calculate Et
            E_t = (np.transpose(belief_dynamics.G[t]) @ S_tplus1 @ belief_dynamics.F[t]) + (np.transpose(belief_dynamics.Gi_1[t]) @ S_tplus1 @ belief_dynamics.Fi_1[t]) + (np.transpose(belief_dynamics.Gi_2[t]) @ S_tplus1 @ belief_dynamics.Fi_2[t])

            # Calculate L_t and l_t (value wrt input and belief)

            L_t = - (np.linalg.inv(D_t) @ E_t)
            valueData.L.insert(0, L_t)

            l_t = - (np.linalg.inv(D_t) @ d_t)
            valueData.l.insert(0, l_t)

            q_t = self._calculate_gradient_belief_stage_cost(belief, input_i)

            # Calculate Ct and c_t (value wrt belief changes)
            C_t = self.Q_t + (np.transpose(belief_dynamics.F[t]) @ S_tplus1 @ belief_dynamics.F[t]) + (np.transpose(belief_dynamics.Fi_1[t]) @ S_tplus1 @ belief_dynamics.Fi_1[t]) + (np.transpose(belief_dynamics.Fi_2[t]) @ S_tplus1 @ belief_dynamics.Fi_2[t])

            c_t = q_t + (np.transpose(belief_dynamics.F[t]) @ s_tplus1_vec) + (np.transpose(belief_dynamics.Fi_1[t]) @ S_tplus1 @ belief_dynamics.ei_1[t]) + (np.transpose(belief_dynamics.Fi_2[t]) @ S_tplus1 @ belief_dynamics.ei_2[t])

            # Calculate quadratized value matrix and gradient for the next iteration
            
            S_t = C_t - (np.transpose(E_t) @ (np.linalg.inv(D_t) @ E_t))

            s_tplus1_vec = c_t - (np.transpose(E_t) @ (np.linalg.inv(D_t) @ d_t))
            S_tplus1 = S_t
        
        return valueData

    def calculate_trajectory_cost(self, beliefs, inputs, belief_dynamics, valueData, goal):

        # Given a trajectory (beliefs and inputs) and its belief dynamics and value matrices, calculate the expected cost of 
        # the trajectory. The cost is given by the stage costs at each time step and the terminal cost of reaching the goal.

        # Calculate terminal cost
 
        S_tplus1 = self.Q_l
        terminal_belief = beliefs[-1]
        s_tplus1 = self._calculate_terminal_cost(terminal_belief, goal)

        for i in reversed(range(len(inputs))):
            belief = beliefs[i]
            input_i = inputs[i]
            cost = self._calculate_stage_cost(belief, input_i)

            ei_1 = belief_dynamics.ei_1[i]
            ei_2 = belief_dynamics.ei_2[i]

            # Calculate stage cost and update the expected cost value matrix

            stage_addition = np.real((cost + ((1/2) * ((np.transpose(ei_1) @ S_tplus1 @ ei_1) + (np.transpose(ei_2) @ S_tplus1 @ ei_2)))))[0][0]
            s_tplus1 += stage_addition

            L_t = valueData.L[i]

            mat_1 = (belief_dynamics.F[i] + (belief_dynamics.G[i] @ L_t))
            mat_2 = (belief_dynamics.Fi_1[i] + (belief_dynamics.Gi_1[i] @ L_t))
            mat_3 = (belief_dynamics.Fi_2[i] + (belief_dynamics.Gi_2[i] @ L_t))
            S_tplus1 = self.Q_t + (np.transpose(L_t) @ self.R_t @ L_t) + (np.transpose(mat_1) @ S_tplus1 @ mat_1) + (np.transpose(mat_2) @ S_tplus1 @ mat_2) + (np.transpose(mat_3) @ S_tplus1 @ mat_3)
        
        return np.real(s_tplus1)

    def get_new_path(self, beliefs, old_u, step_size, valueData):

        # Given a trajectory (beliefs and inputs) and the value gradients computed at each stage, compute a new path that can 
        # minimize the uncertainity along the path. This is done be applying gradient descent.
        # The step size is an input to this method!

        new_u = list()
        
        new_path = list()
        new_path.append(beliefs[0])

        cur_belief = beliefs[0]

        for i in range(len(old_u)): 
            u_bar = old_u[i]
            b_bar = beliefs[i]

            C = self.estimator.C[i]
            A = self.estimator.A[i]
            M = self.estimator.M[i]

            x = cur_belief[0:2].reshape((2,1))
            cov = cur_belief[2:]
            cov = cov.reshape((2,2))

            measurement, N = self.environment.get_measurement(x.flatten())

            new_u_val = u_bar + ((valueData.L[i] @ (cur_belief - b_bar)) + (step_size * valueData.l[i]))
            new_u.append(new_u_val)

            new_b, _, _, _, _, _ = self.estimator.apply_dynamics_and_estimate_tplus1(A, C, M, N, cov, x, measurement, new_u_val)
            new_path.append(new_b)
            cur_belief = new_b

        new_path = np.real(new_path)
        new_u = np.real(new_u)
        return [new_path, new_u]

    """
    Private Functions
    """

    def _calculate_terminal_cost(self, terminal_belief, goal):

        # Calculate terminal cost using the terminal belief of a given path and the intended goal
        
        ideal_end_belief = np.zeros((6,1))
        ideal_end_belief[0] = goal[0]
        ideal_end_belief[1] = goal[1]
        belief_diff = terminal_belief - ideal_end_belief

        cost = (np.transpose(belief_diff) @ self.Q_l @ belief_diff) 
        return cost[0][0]

    def _calculate_stage_cost(self, belief, input_i):

        # Calculate stage cost using a given belief in a path and the input at that time step

        cov = belief[2:]
        trace = (np.transpose(cov) @ cov) # Ignoring Q_t as it is identity
        cost = trace + (np.transpose(input_i) @ self.R_t @ input_i)
        return cost[0][0]

    def _calculate_gradient_input_stage_cost(self, belief, input_i):

        # Calculate gradient of the stage cost wrt the input effort

        h = self.h

        r_t = []

        for j in range(len(input_i)):

            input_1 = input_i.copy()
            input_2 = input_i.copy()

            input_1[j] += h
            input_2[j] -= h

            cost_1 = self._calculate_stage_cost(belief, input_1)
            cost_2 = self._calculate_stage_cost(belief, input_2)

            cost_diff = (cost_1 - cost_2) / (2*h)
            r_t.append(cost_diff)
        
        r_t = np.hstack(r_t).reshape((len(input_i),1))

        return r_t

    def _calculate_gradient_belief_stage_cost(self, belief, input_i):

        # Calculate gradient of the stage cost wrt the belief of the robot

        h = self.h

        q_t = []

        for j in range(len(belief)):

            belief_1 = belief.copy()
            belief_2 = belief.copy()

            belief_1[j] += h
            belief_2[j] -= h

            cost_1 = self._calculate_stage_cost(belief_1, input_i)
            cost_2 = self._calculate_stage_cost(belief_2, input_i)

            cost_diff = (cost_1 - cost_2) / (2*h)
            q_t.append(cost_diff)
        
        q_t = np.hstack(q_t).reshape((len(belief),1))

        return q_t
    
    def _calculate_terminal_cost_gradient(self, belief_l):

        # Calculate gradient of the terminal cost at the last belief state

        h = self.h

        s_tplus1_vec = []

        for j in range(len(belief_l)):
            b_1 = belief_l.copy()
            b_2 = belief_l.copy()

            b_1[j] += h
            b_2[j] -= h

            c1 = self._calculate_terminal_cost(b_1, belief_l)
            c2 = self._calculate_terminal_cost(b_2, belief_l)

            c_diff = (c1 - c2) / (2*h)
            s_tplus1_vec.append(c_diff)
        
        s_tplus1_vec = np.vstack(s_tplus1_vec)

        return s_tplus1_vec

    
    def _calculate_belief_gradients(self, belief, input_i, i):

        # Calculate gradient of the belief dynamics wrt changes to the beliefs at a given time step

        h = self.h

        C = self.estimator.C[i]
        A = self.estimator.A[i]
        M = self.estimator.M[i]

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

            g_1, w1_1, w1_2, _, _, _ = self.estimator.apply_dynamics_and_estimate_tplus1(A, C, M, N_1, cov_1, x_1, measurement_1, input_i)
            g_2, w2_1, w2_2, _, _, _ = self.estimator.apply_dynamics_and_estimate_tplus1(A, C, M, N_2, cov_2, x_2, measurement_2, input_i)

            g_diff = (g_1 - g_2) / (2*h)
            w1_diff = (w1_1 - w2_1) / (2*h)
            w2_diff = (w1_2 - w2_2) / (2*h)

            Fit_1.append(w1_diff)
            Fit_2.append(w2_diff)
            Ft.append(g_diff)
        
        Ft = np.hstack(Ft)
        Fit_1 = np.hstack(Fit_1)
        Fit_2 = np.hstack(Fit_2)
        
        return (Ft, Fit_1, Fit_2)

    def _calculate_input_gradients(self, belief, input_i, i):

        # Calculate gradient of the belief dynamics wrt changes to the inputs at a given time step

        h = self.h

        C = self.estimator.C[i]
        A = self.estimator.A[i]
        M = self.estimator.M[i]

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

            g_1, w1_1, w1_2, _, _, _ = self.estimator.apply_dynamics_and_estimate_tplus1(A, C, M, N, cov, x, measurement, input_1)
            g_2, w2_1, w2_2, _, _, _= self.estimator.apply_dynamics_and_estimate_tplus1(A, C, M, N, cov, x, measurement, input_2)

            g_diff = (g_1 - g_2) / (2*h)
            w1_diff = (w1_1 - w2_1) / (2*h)
            w2_diff = (w1_2 - w2_2) / (2*h)

            Git_1.append(w1_diff)
            Git_2.append(w2_diff)
            Gt.append(g_diff)

        Gt = np.hstack(Gt)
        Git_1 = np.hstack(Git_1)
        Git_2 = np.hstack(Git_2)
        
        return (Gt, Git_1, Git_2)

