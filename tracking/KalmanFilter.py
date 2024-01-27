import numpy as np


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt
        self.u = np.matrix([[u_x], [u_y]])
        self.std_acc = std_acc
        self.x_std_meas = x_std_meas
        self.y_std_meas = y_std_meas
        self.x = np.matrix([[0], [0], [0], [0]])

        self.A = np.matrix(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        self.B = np.matrix(
            [
                [(self.dt**2) / 2, 0],
                [0, (self.dt**2) / 2],
                [self.dt, 0],
                [0, self.dt],
            ]
        )

        self.H = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]])

        self.Q = (
            np.matrix(
                [
                    [(self.dt**4) / 4, 0, (self.dt**3) / 2, 0],
                    [0, (self.dt**4) / 4, 0, (self.dt**3) / 2],
                    [(self.dt**3) / 2, 0, self.dt**2, 0],
                    [0, (self.dt**3) / 2, 0, self.dt**2],
                ]
            )
            * self.std_acc**2
        )

        self.R = np.matrix([[self.x_std_meas**2, 0], [0, self.y_std_meas**2]])

        self.P = np.eye(self.A.shape[1])

    def predict(self):
        self.x = self.A @ self.x + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2]

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = np.round(self.x + K @ (z - self.H @ self.x))

        self.P = (np.eye(self.H.shape[1]) - K @ self.H) @ self.P

        return self.x[:2]
