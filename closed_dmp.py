from dmpling.dmp import DMP
import numpy as np

# works just like dmpling's dmp, except step functions off of external
# pose than just following internal pose open-loop

class ClosedDMP(DMP):
    def __init__(self, T, dt, a=150, b=25, n_bfs=10):
        super().__init__(T, dt, a=a, b=b, n_bfs=n_bfs)

    def step(self, y=None, tau=1.0, k=1.0, start=None, goal=None):
        if y is not None:
            self.y = y
        return super().step(tau=tau, k=k, start=start, goal=goal)