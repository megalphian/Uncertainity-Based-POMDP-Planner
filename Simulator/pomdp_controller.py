import numpy as np

class pomdp_controller:

    def __init__(self, optimal_path, S_l):

        self.b_bar = optimal_path
        self.u_bar = list()

        self.compute_u_bar()

        self.Q_t = np.identity(2)
        self.R_t = np.identity(2)
        self.Q_l = 10 * len(b_bar) * np.identity(2)

    def compute_u_bar(self):

        for i in range(len(self.b_bar) - 1):
            b_i = self.b_bar[i]
            b_j = self.b_bar[i+1]
            
            u_x = b_j[0] - b_i[0]
            u_y = b_j[1] - b_i[1]

            self.u_bar.append([u_x, u_y])
