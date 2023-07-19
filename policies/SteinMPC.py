import numpy as np


class SteinMPC:
    def __init__(
        self,
        means,
        cov,
        num_particles,
        num_samples_per_particle,
        cost_likelihood_funct,
        dynamics,
    ) -> None:
        self.means = means
        self.cov = cov  # Should probably change to allow for multiple covariances...

        self.cost_likelihood_funct = cost_likelihood_funct
        self.num_particles = num_particles
        self.num_samples_per_particle = num_samples_per_particle

        self.dynamics = dynamics
        self.dim_u = dynamics.dim_u
        self.dim_x = dynamics.dim_x

    def setup_run(self, x0, time_horizon):
        self.state = x0
        self.time_horizon = time_horizon

    def sample_particles(self):
        return np.random.multivariate_normal(
            np.zeros(self.dim_u),
            self.cov,
            size=(self.num_particles, self.time_horizon, self.num_samples_per_particle),
        ) + np.tile(self.means, (1, 1, self.num_samples_per_particle, self.dim_u))
