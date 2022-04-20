import numpy as np


def CIR(
        T: int = 30,
        r0: float = 0.078,
        kappa: float = 0.6,
        sigma: float = 0.12,
        r_bar: float = 0.08,
        simulations: int = 100
) -> np.ndarray:
    ndays = int(T * 360)  # days
    dt = 1 / 360.

    cir_mat = np.zeros((simulations, ndays + 1))
    cir_mat[:, 0] = r0
    for i in range(1, ndays + 1):
        brownian = np.random.standard_normal(simulations)
        cir_mat[:, i] = cir_mat[:, i - 1] + kappa * (r_bar - np.maximum(cir_mat[:, i - 1], 0)) * dt + \
                        sigma * brownian * np.sqrt(dt * np.maximum(cir_mat[:, i - 1], 0))
    return cir_mat


def ZCB_CIR(tau, kappa, sigma, r_bar, r_t1):
    h1 = np.sqrt(kappa ** 2 + 2 * sigma ** 2)
    h2 = (kappa + h1) / 2.
    h3 = 2 * kappa * r_bar / sigma ** 2

    B = (np.exp(h1 * tau) - 1) / (h2 * (np.exp(h1 * tau) - 1) + h1)
    A = np.power(h1 * np.exp(h2 * tau) / (h2 * (np.exp(h1 * tau) - 1) + h1), h3)
    return A * np.exp(-B * r_t1)
