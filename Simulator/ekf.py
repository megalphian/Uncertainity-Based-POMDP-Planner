from scipy.linalg import sqrtm
import numpy as np

class KalmanEstimator:

    def __init__(self, environment, time_step, step_count, m_multiplier):
        self.x_est = None
        self.cov_est = None

        self.belief_states = None
        self.W1 = None
        self.W2 = None

        self.init_system_matrices(step_count, m_multiplier)
        self.environment = environment
        self.time_step = time_step

    def make_EKF_Estimates(self, init_est, inputs, init_cov):
        cov = sqrtm(init_cov)
        x = np.asarray([[init_est[0]], [init_est[1]]])
        x_actual = x

        self.x_est = [x]
        self.cov_est = [cov]
        self.belief_states = [np.concatenate((x.flatten(), cov.flatten())).reshape((6,1))]
        self.W1 = [np.zeros((1,6))]
        self.W2 = [np.zeros((1,6))]

        for i in range(len(inputs)):

            C = self.C[i]
            A = self.A[i]
            M = self.M[i]

            measurement, N = self.environment.get_measurement(x_actual.flatten())
            input_i = inputs[i]

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
        
        input_steps = input_i * self.time_step
        x_actual = A @ x + input_steps

        x = x_actual + (K @ (measurement - (C @ x)))

        belief = np.concatenate((x_actual.flatten(), cov.flatten()))
        belief = belief.reshape((6,1))

        zeros = np.zeros((1,4))
        w1 = np.concatenate((w_term[:,0].flatten(), zeros.flatten()))
        w1 = w1.reshape((6,1))

        w2 = np.concatenate((w_term[:,1].flatten(), zeros.flatten()))
        w2 = w2.reshape((6,1))

        return (belief, w1, w2, x, cov, x_actual)
    
    def init_system_matrices(self, iterations, m_multiplier):
        self.A = list()
        self.C = list()
        self.M = list()

        for i in range(iterations):
            self.A.append(np.identity(2))
            self.M.append(m_multiplier * np.identity(2))
            self.C.append(np.identity(2))