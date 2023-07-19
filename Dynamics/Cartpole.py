import numpy as np
import math


class Cartpole:
    def __init__(self, mQ, mC, l, g, dim_x, dim_u) -> None:
        self.mQ = mQ
        self.mC = mC
        self.l = l
        self.g = g

        self.M = self.mQ + self.mC
        self.ml = self.mQ * self.l

        self.dim_x = dim_x
        self.dim_u = dim_u

    def setCostFunction(self, step_cost_funct, terminal_cost_funct):
        self.step_cost_funct = step_cost_funct
        self.terminal_cost_funct = terminal_cost_funct

    def setNoise(self, noise_cov, noise_mean):
        self.noise_cov = noise_cov
        self.noise_mean = noise_mean

    def simulateTimestep(self, prev_states: np.ndarray, u_samples: np.ndarray):
        # For readability (and speed, probably?)
        x = prev_states[:, 0]
        dx = prev_states[:, 1]
        theta = prev_states[:, 2]
        dtheta = prev_states[:, 3]

        # Get trig functions of our theta-values, since we use them a lot
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)

        # Generate second derivatives
        d2theta = (
            (self.M * self.g * sinTheta)
            - (self.ml * dtheta * dtheta * sinTheta * cosTheta)
            - (u_samples * cosTheta)
        )
        d2theta = d2theta / (
            (self.M * self.l * 4 / 3) - (self.ml * cosTheta * cosTheta)
        )
        d2x = (
            u_samples
            + (self.ml * dtheta * dtheta * sinTheta)
            - (self.ml * d2theta * cosTheta)
        )
        d2x = d2x / self.M

        # Concatinate back into a single matrix of states
        nextStates = np.empty_like(prev_states)
        nextStates[:, 0] = x + (dx * self.T)
        nextStates[:, 1] = dx + (d2x * self.T)
        nextStates[:, 2] = (
            np.mod(math.pi + theta + (dtheta * self.T), 2 * math.pi) - math.pi
        )
        nextStates[:, 3] = dtheta + (d2theta * self.T)

        return nextStates
