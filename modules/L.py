import numpy as np

import modules.IR as IR


def generate_CPR(
        wac=0.1,
        T=30,
        notional=100000,
        r0=0.078,
        kappa=0.6,
        r_bar=0.08,
        sigma=0.12,
        simulations=5000
) -> np.ndarray:
    period = int(T * 12)  # months
    dt = 1 / 360.
    rm = wac / 12.

    PV = np.zeros((simulations, period + 1))
    PV[:, 0] = notional
    c = np.zeros((simulations, period))
    SY = [0.94, 0.76, 0.74, 0.95, 0.98, 0.92, 0.98, 1.10, 1.18, 1.22, 1.23, 0.98]

    cir_mat = IR.CIR(T, r0, kappa, sigma, r_bar, simulations)

    for i in range(period):
        # compute CPR
        BU = 0.3 + 0.7 * (PV[:, i] / PV[:, 0])
        SG = min(1, (i + 1) / 30.)
        SY_i = SY[i % 12]
        P10 = IR.ZCB_CIR(10, kappa, sigma, r_bar, cir_mat[:, i * 30])
        r10 = -1 / 10. * np.log(P10)
        RI = 0.28 + 0.14 * np.arctan(-8.57 + 430 * (wac - r10))
        CPR = BU * SG * SY_i * RI

        IP = PV[:, i] * rm
        SP = PV[:, i] * rm * (1 / (1 - pow(1 + rm, -period + i)) - 1.)
        PP = (PV[:, i] - SP) * (1 - pow(1 - CPR, 1 / 12.))
        PV[:, i + 1] = PV[:, i] - SP - PP
        c[:, i] = CPR

    step = 30 * np.arange(1, period + 1, 1)
    R_cir = np.array([dt * np.sum(cir_mat[:, 1:j], axis=1) for j in step]).T
    disc = np.exp(-R_cir)
    disc_c = disc * c
    return disc_c
