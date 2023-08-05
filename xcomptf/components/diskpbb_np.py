"""dikspbb translated from FORTRAN code in Xspec.

This module is translated from ChatGPT3.5 in Aug. 5th, 2023
by Tomoki Omama. Some modification and comment adding has been
done to enhance the readibity.
The user should be noted that this code is not well examined
and thus, sometimes output wrong result from the original ones.
"""
import numpy as np


def dkbflx(TIN, p, E, dtype=np.float32):
    """
    Calculate the flux of the disk blackbody model at a given energy.

    Args:
        TIN (float): Inner temperature of the disk (keV).
        p (float): Exponent for the temperature radial
            dependence (T(r) ~ r^-p).
        E (float): Energy (keV).

    Returns:
        float: Photon flux in photons/s/cm^2/keV.
    """
    GAUSS = np.array(
        [0.236926885, 0.478628670, 0.568888888, 0.478628670,
         0.236926885, -0.906179846, -0.538469310, 0.0,
         0.538469310, 0.906179846], dtype=dtype)

    NNN = 10000
    if E > TIN:
        NNN = 5
    elif E > 0.2 * TIN:
        NNN = 10
    elif E > 0.1 * TIN:
        NNN = 100
    elif E > 0.01 * TIN:
        NNN = 500
    elif E > 0.001 * TIN:
        NNN = 1000

    XN = 1.0 / NNN / 2.0
    PHOTON = 0.0

    for i in range(1, NNN + 1):
        XH = XN * (2 * i - 1)
        for j in range(5):
            X = XN * GAUSS[j + 5] + XH
            if E / TIN / X >= 170.0:
                DK = X ** (-2.0 / p - 1.0) * np.exp(-E / TIN / X)
            else:
                DK = X ** (-2.0 / p - 1.0) / (np.exp(E / TIN / X) - 1.0)

            PHOTON += GAUSS[j] * DK * XN

    PHOTON = 2.78e-3 * E * E * PHOTON * (0.75 / p)

    return PHOTON


def diskpbb(EAR, NE, PARAM, IFL, dtype=np.float32):
    """
    Compute the multicolor disk blackbody model spectrum.

    Args:
        EAR (array-like): Energy bin edges (keV).
        NE (int): Number of energy bins.
        PARAM (array-like): Model parameters [TIN (keV), p].
        IFL (int): Flag for error computation (not used in this model).

    Returns:
        tuple: Tuple containing arrays of photon counts in each energy
            bin (PHOTAR) and errors (PHOTER).
    """
    GAUSS = np.array(
        [0.236926885, 0.478628670, 0.568888888, 0.478628670,
         0.236926885, -0.906179846, -0.538469310, 0.0,
         0.538469310, 0.906179846], dtype=dtype)

    PHOTER = np.zeros(NE, dtype=dtype)

    TIN, p = PARAM

    EAR_shifted = EAR[1:]
    EAR_prev = EAR[:-1]
    XN = (EAR_shifted - EAR_prev) / 2.0

    PHOTAR = np.zeros(NE, dtype=dtype)
    for i in range(NE):
        XH = XN[i] + EAR_prev[i]
        photon = 0.0
        for j in range(5):
            E = XN[i] * GAUSS[j + 5] + XH
            photon += GAUSS[j] * dkbflx(TIN, p, E)
        PHOTAR[i] = photon * XN[i]

    return PHOTAR, PHOTER


def main():
    NE = 3491                          # NE: Number of Energies?
    EAR = np.linspace(0.1, 20., NE+1)  # EAR: Energy ARray?
    PARAM = [0.2, 3/4]                 # [Tin, p]
    IFL = 1                            # ???

    import time
    t0 = time.time()
    PHOTAR, PHOTER = diskpbb(EAR, NE, PARAM, IFL)
    t1 = time.time()
    print("ran time is {} s".format(t1-t0))
    print(PHOTAR)

    import matplotlib.pyplot as plt
    energy_centers = (EAR[:-1] + EAR[1:]) / 2
    fig, ax = plt.subplots()
    ax.plot(energy_centers, PHOTAR)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
