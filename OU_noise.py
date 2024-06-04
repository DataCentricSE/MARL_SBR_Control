import numpy as np

class OUActionNoise():
    def __init__(self, mu, sigma=0.2, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0

        self.reset() #to reset the noise et every top of episode

    def __call__(self):
        # noise = OUActionNOise()
        # current_noise = noise() That 's what __call__ can do

        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def exploration_decay(self):
        if self.sigma <= 0.01:
            return
        self.sigma -= 0.0001

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        
class NormalActionNoise():
    def __init__(self, mu=0, sigma=0.15):
        self.mu = mu
        self.sigma = sigma
    def __call__(self):
        noise = np.random.normal(loc=self.mu, scale=self.sigma)
        return  noise

    def exploration_decay(self):
        if self.sigma <= 0.2:
            return
        self.sigma -= 0.0025



