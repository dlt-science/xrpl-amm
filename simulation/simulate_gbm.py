import numpy as np


class SimulateGBM:
    def simulate_GBM(self, T, mu, sigma, S0, dt):
        N = int(T / dt)
        t = np.linspace(0, T, N, endpoint=False)
        W = np.random.standard_normal(size=N)
        W = np.cumsum(W) * np.sqrt(dt)
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        S = S0 * np.exp(X)
        return S

    def GBM_with_precise_fixed_point(self, x, T, mu, sigma, S0, dt):
        """
        Simulate a continuous GBM path and adjust it to ensure the value at index x is S0.
        """
        # ssimulate a GBM path over the entire interval
        S_continuous = self.simulate_GBM(T, mu, sigma, S0, dt)

        adjustment_factor = S0 / S_continuous[x]
        S_adjusted = S_continuous * adjustment_factor

        return S_adjusted
