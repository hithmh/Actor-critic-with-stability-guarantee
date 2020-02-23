import sparsegrad.forward as forward
import numpy as np
import osqp
import scipy.sparse as sparse
import cvxpy as cp


class MPC(object):
    def __init__(self, a_dim, s_dim, variant):
        self.horizon = variant['horizon']
        theta_threshold_radians = 20 * 2 * math.pi / 360
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.s_bound = variant['s_bound']
        self.a_bound = variant['a_bound']
        length = 0.5
        masscart = 1
        masspole = 0.1
        total_mass = (masspole + masscart)
        polemass_length = (masspole * length)
        g = 10
        H = np.array([
            [1, 0, 0, 0],
            [0, total_mass, 0, - polemass_length],
            [0, 0, 1, 0],
            [0, - polemass_length, 0, (2 * length) ** 2 * masspole / 3]
        ])

        Hinv = np.linalg.inv(H)

        self.A = Hinv @ np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, - polemass_length * g, 0]
        ])
        self.B = Hinv @ np.array([0, 1.0, 0, 0]).reshape((4, 1))
        self.Q = np.diag([1/100, 0., 20 *(1/ theta_threshold_radians)**2, 0.])
        self.R = np.array([[0.1]])



    def choose_action(self, x_0, arg):

        constraints = []
        X = [cp.Variable(self.s_dim) for _ in range(self.horizon)]
        U = [cp.Variable(self.a_dim) for _ in range(self.horizon)]
        obj = 0
        constraints.append(X[0]==x_0)
        for i in range(self.horizon):
            constraints.extend(
                [X[i+1] == self.A*X[i] + self.B*U[i],
                 X[i] <= self.s_bound.high,
                 X[i] >= self.s_bound.low,
                 U[i] <= self.a_bound.high,
                 U[i] >= self.a_bound.low,])
            obj = obj + X[i].T * self.Q *X[i] + U[i].T * self.R * U[i]
        obj = cv.Minimize(obj)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        return U[0]

    def restore(self, log_path):

        return True