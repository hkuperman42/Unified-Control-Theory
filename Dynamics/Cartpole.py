import numpy as np
import math

class Cartpole():
    def __init__(self, mQ, mC, l, g) -> None:
        self.mQ = mQ;
        self.mC = mC
        self.l = l;
        self.g = g;
        
        self.M = self.mQ + self.mC
        self.ml = self.mQ * self.l
    
    
    def simulateTimestep(self, prevStates : np.ndarray, uSamples : np.ndarray):
        #For readability (and speed, probably?)
        x      = prevStates[:, 0]
        dx     = prevStates[:, 1]
        theta  = prevStates[:, 2]
        dtheta = prevStates[:, 3]

        #Get trig functions of our theta-values, since we use them a lot
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)

        #Generate second derivatives
        d2theta = (self.M * self.g * sinTheta) - (self.ml * dtheta * dtheta * sinTheta * cosTheta) - (uSamples * cosTheta)
        d2theta = d2theta / ((self.M * self.l * 4 / 3) - (self.ml * cosTheta * cosTheta))
        d2x = uSamples + (self.ml * dtheta * dtheta * sinTheta) - (self.ml * d2theta * cosTheta)
        d2x = d2x / self.M

        #Concatinate back into a single matrix of states
        nextStates = np.empty_like(prevStates)
        nextStates[:, 0] = x + (dx * self.T)
        nextStates[:, 1] = dx + (d2x * self.T)
        nextStates[:, 2] = np.mod(math.pi + theta + (dtheta * self.T), 2 * math.pi) - math.pi
        nextStates[:, 3] = dtheta + (d2theta * self.T)

        return nextStates