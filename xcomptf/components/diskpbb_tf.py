"""dikspbb translated from fortran code in xspec.

this module is translated from chatgpt3.5 in aug. 5th, 2023
by tomoki omama. some modification and comment adding has been
done to enhance the readibity.
the user should be noted that this code is not well examined
and thus, sometimes output wrong result from the original ones.
"""
import tensorflow as tf


def dkbflx(tin, photon_index, energy, dtype=tf.float32):
    """
    calculate the flux of the disk blackbody model at a given energy.

    args:
        tin (float): inner temperature of the disk (kev).
        p (float): exponent for the temperature radial
            dependence (t(r) ~ r^-p).
        energy (float): energy (kev).

    returns:
        float: photon flux in photons/s/cm^2/kev.
    """
    gauss = tf.constant(
        [0.236926885, 0.478628670, 0.568888888, 0.478628670,
         0.236926885, -0.906179846, -0.538469310, 0.0,
         0.538469310, 0.906179846], dtype=dtype)

    nnn = 1000
    xn = 1.0 / tf.cast(nnn, dtype) / 2.0

    gauss_left = gauss[0:5]
    gauss_right = gauss[5:10]

    xh = xn * (2. * tf.range(1, nnn+1, 1, dtype=dtype) - 1)
    x = xn * gauss_right[tf.newaxis, :] + xh[..., tf.newaxis]

    energy_batch = energy[..., tf.newaxis, tf.newaxis]
    dk = tf.where(
        energy_batch / tin / x >= 170,
        x ** (-2.0 / photon_index - 1.0) * tf.exp(
            -energy_batch / tin / x),
        x ** (-2.0 / photon_index - 1.0) / (
            tf.exp(energy_batch / tin / x) - 1.0)
    )
    photon = tf.reduce_sum(gauss_left * dk * xn, axis=[-2, -1])
    photar = 2.78e-3 * energy * energy * photon * (0.75 / photon_index)

    assert x.shape == (nnn, 5)
    assert energy.shape == (3491, 5)
    assert dk.shape == (3491, 5, nnn, 5)
    assert gauss_left.shape == (5,)
    assert photon.shape == (3491, 5)
    assert photar.shape == (3491, 5)

    return photar


@tf.function(autograph=False, jit_compile=True)
def diskpbb(ear, ne, param, ifl, dtype=tf.float32):
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
    gauss = tf.constant(
        [0.236926885, 0.478628670, 0.568888888, 0.478628670,
         0.236926885, -0.906179846, -0.538469310, 0.0,
         0.538469310, 0.906179846], dtype=dtype)

    photer = tf.zeros(ne, dtype=dtype)

    tin, p = param

    ear_shifted = ear[1:]
    ear_prev = ear[:-1]
    xn = (ear_shifted - ear_prev) / 2.0

    xh = xn + ear_prev
    gauss_left = gauss[0:5]
    gauss_right = gauss[5:10]
    energy = xn[:, tf.newaxis] * gauss_right[..., tf.newaxis, :] + \
        xh[:, tf.newaxis]

    photon = tf.reduce_sum(
        gauss_left[..., tf.newaxis, :] * dkbflx(tin, p, energy),
        axis=-1)
    photar = photon * xn

    return photar, photer


def main():
    ne = 3491                          # ne: number of energies?
    ear = tf.linspace(0.1, 20., ne+1)  # ear: energy array?
    param = [0.2, 3/4]                 # [tin, p]
    ifl = 1                            # ???

    import time
    t0 = time.time()
    photar, photer = diskpbb(ear, ne, param, ifl)
    t1 = time.time()
    print("ran time is {} s".format(t1-t0))
    print(photar)

    import matplotlib.pyplot as plt
    energy_centers = (ear[:-1] + ear[1:]) / 2
    fig, ax = plt.subplots()
    ax.plot(energy_centers, photar)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
