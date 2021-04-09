from scipy.linalg import sqrtm
import numpy as np

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

            measurement, N, N_diff = self.environment.get_measurement(x_actual.flatten())
            input_i = np.asarray([[inputs[i][0]], [inputs[i][1]]])

            belief, w1, w2, x, cov, x_actual = self.make_estimate(A, C, M, N, N_diff, cov, x, measurement, input_i)

            self.belief_states.append(belief)

            self.W1.append(w1)
            self.W2.append(w2)

            self.x_est.append(x)
            self.cov_est.append(cov)

    def make_estimate(self, A, C, M, N, N_diff, cov, x, measurement, input_i):

        A_cov = A @ cov
        tau = (A_cov) @ (np.transpose(A_cov)) + (M @ np.transpose(M))

        K = (tau @ np.transpose(C)) @ np.linalg.inv((C @ tau @ np.transpose(C)) + (N @ np.transpose(N)))

        w_term = sqrtm(K @ C @ tau)
        cov = sqrtm(tau - (K @ C @ tau))
        
        input_steps = [u * self.time_step for u in input_i]
        x_actual = A @ x + input_steps

        belief = np.concatenate((x_actual.flatten(), cov.flatten()))

        x = x_actual + (K @ (measurement - (C @ x)))

        zeros = np.zeros((1,4))
        w1 = np.concatenate((w_term[:,0].flatten(), zeros.flatten()))

        w2 = np.concatenate((w_term[:,1].flatten(), zeros.flatten()))

        return (belief, w1, w2, x, cov, x_actual)