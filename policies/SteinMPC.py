import numpy as np

class SteinMPC():
    def __init__(self, means, sigmas, num_particles, num_samples_per_particle, cost_likelihood_funct) -> None:
        self.means = means;
        self.sigmas = sigmas;
        
        self.cost_likelihood_funct = cost_likelihood_funct;
        self.num_particles = num_particles;
        self.num_samples_per_particle = num_samples_per_particle;
        
    def setup_run(self, x0, time_horizon):
        self.state = x0
        self.time_horizon = time_horizon
        
    
        
        
    
    
