"""dikspbb translated from fortran code in xspec.

this module is translated from chatgpt3.5 in aug. 5th, 2023
by tomoki omama. some modification and comment adding has been
done to enhance the readibity.
the user should be noted that this code is not well examined
and thus, sometimes output wrong result from the original ones.
"""
import numpy as np


def dkbflx(tin, p, e, dtype=np.float32):
    """
    calculate the flux of the disk blackbody model at a given energy.

    args:
        tin (float): inner temperature of the disk (kev).
        p (float): exponent for the temperature radial
            dependence (t(r) ~ r^-p).
        e (float): energy (kev).

    returns:
        float: photon flux in photons/s/cm^2/kev.
    """
    gauss = np.array(
        [0.236926885, 0.478628670, 0.568888888, 0.478628670,
         0.236926885, -0.906179846, -0.538469310, 0.0,
         0.538469310, 0.906179846], dtype=dtype)

    nnn = 10000
    if e > tin:
        nnn = 5
    elif e > 0.2 * tin:
        nnn = 10
    elif e > 0.1 * tin:
        nnn = 100
    elif e > 0.01 * tin:
        nnn = 500
    elif e > 0.001 * tin:
        nnn = 1000

    xn = 1.0 / nnn / 2.0
    photon = 0.0

    for i in range(1, nnn + 1):
        xh = xn * (2 * i - 1)
        for j in range(5):
            x = xn * gauss[j + 5] + xh
            if e / tin / x >= 170.0:
                dk = x ** (-2.0 / p - 1.0) * np.exp(-e / tin / x)
            else:
                dk = x ** (-2.0 / p - 1.0) / (np.exp(e / tin / x) - 1.0)

            photon += gauss[j] * dk * xn

    photon = 2.78e-3 * e * e * photon * (0.75 / p)

    return photon


def diskpbb(ear, ne, param, ifl, dtype=np.float32):
    """
    compute the multicolor disk blackbody model spectrum.

    args:
        ear (array-like): energy bin edges (kev).
        ne (int): number of energy bins.
        param (array-like): model parameters [tin (kev), p].
        ifl (int): flag for error computation (not used in this model).

    returns:
        tuple: tuple containing arrays of photon counts in each energy
            bin (photar) and errors (photer).
    """
    gauss = np.array(
        [0.236926885, 0.478628670, 0.568888888, 0.478628670,
         0.236926885, -0.906179846, -0.538469310, 0.0,
         0.538469310, 0.906179846], dtype=dtype)

    photer = np.zeros(ne, dtype=dtype)

    tin, p = param

    ear_shifted = ear[1:]
    ear_prev = ear[:-1]
    xn = (ear_shifted - ear_prev) / 2.0

    photar = np.zeros(ne, dtype=dtype)
    for i in range(ne):
        xh = xn[i] + ear_prev[i]
        photon = 0.0
        for j in range(5):
            e = xn[i] * gauss[j + 5] + xh
            photon += gauss[j] * dkbflx(tin, p, e)
        photar[i] = photon * xn[i]

    return photar, photer


def main():
    ne = 3491                          # ne: number of energies?
    ear = np.linspace(0.1, 20., ne+1)  # ear: energy array?
    param = [0.2, 3/4]                 # [tin, p]
    ifl = 1                            # ???

    np.random.seed(1)
    params = np.random.normal(loc=[0.2, 0.75], scale=[0.01, 0.0],
                              size=(10000, 2))
    print("parameter shape is {}".format(params.shape))
    print(params)

    import time
    from tqdm import tqdm
    t0 = time.time()
    photars = []
    for param in tqdm(params):
        photar, photer = diskpbb(ear, ne, param, ifl)
        photars.append(photar)
    t1 = time.time()
    print("ran time is {} s".format(t1-t0))
    print(photars)

    import matplotlib.pyplot as plt
    energy_centers = (ear[:-1] + ear[1:]) / 2
    fig, ax = plt.subplots()
    for i in range(params.shape[0]):
        ax.plot(energy_centers, photars[i])
    ax.plot(energy_centers, photar)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
