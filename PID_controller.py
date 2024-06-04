class PID():
    def __init__(self, K=1, TI=0):
        self.K = K
        self.TI = TI

    def update(self, error):
        du = self.K*(error[1] - error[0]) + 1/self.TI*error[1]
        return du