import numpy as np

"""
Things to consider/check:
- Is there a faster ordering of axis?
- Is this usable with a GPU?
- Are the particle weights the average per particle over the cost-likelihood function?
- Should I be staying in log-world for longer
"""

"""
TODO:
- Implement update_particles() lmao
- Add support for different covariances for each particles
- Enable alternate real-control sampling methods in equation (24)
"""


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
        self.cov = cov
        self.cost_likelihood_funct = cost_likelihood_funct
        self.num_particles = num_particles
        self.num_samples_per_particle = num_samples_per_particle
        self.dynamics = dynamics
        self.dim_u = dynamics.dim_u
        self.dim_x = dynamics.dim_x

    def set_cost_functions(
        self,
        control_cost_funct,
        step_cost_funct,
        terminal_cost_funct,
        cost_likelihood_funct,
    ):
        self.control_cost_funct = control_cost_funct
        self.state_cost_funct = step_cost_funct
        self.terminal_cost_funct = terminal_cost_funct
        self.cost_likelihood_funct = cost_likelihood_funct

    def sample_particles(self):
        return np.random.Generator.multivariate_normal(
            np.zeros(self.dim_u),
            self.cov,
            size=(
                self.num_particles,
                self.num_samples_per_particle,
                self.time_horizon,
            ),
        ) + np.tile(self.means, self.num_samples_per_particle).transpose((0, 3, 1, 2))

    def simulate_samples(self, samples):
        states = np.zeros(
            self.num_particles,
            self.num_samples_per_particle,
            self.time_horizon,
            self.dim_x,
        )
        current_states = np.tile(
            self.state, (self.num_particles, self.num_samples_per_particle, 1)
        )
        costs = np.zeros(self.num_particles, self.num_samples_per_particle)

        for i in range(self.time_horizon):
            sampled_controls = samples[:, :, i, :]
            current_states = self.dynamics.simulate_timestep(
                current_states, sampled_controls
            )
            states[:, :, i, :] = current_states
            costs += self.control_cost_funct(sampled_controls) + self.state_cost_funct(
                current_states
            )

        costs += self.terminal_cost_funct(current_states)

        return states, costs

    def apply_cost_likelihood(self, costs):
        cost_weights = self.cost_likelihood_funct(costs)
        particle_weights = np.sum(cost_weights, axis=1)
        particle_weights = np.sum(particle_weights)

        return particle_weights, cost_weights

    def update_particles(self):
        # Do something here
        return 0
