"""Diksbb translated from FORTRAN code in Xspec.

This module is translated from ChatGPT3.5 in Aug. 5th, 2023
by Tomoki Omama. Some modification and comment adding has been
done to enhance the readibity.
The user should be noted that this code is not well examined
and thus, sometimes output wrong result from the original ones.
"""
import tensorflow as tf


def precomputation(dtype=tf.float32):
    return tf.constant(
        [0.96198382E-03, 0.10901181E-02, 0.12310012E-02, 0.13841352E-02,
         0.15481583E-02, 0.17210036E-02, 0.18988943E-02, 0.20769390E-02,
         0.22484281E-02, 0.24049483E-02, 0.25366202E-02, 0.26316255E-02,
         0.26774985E-02, 0.26613059E-02, 0.25708784E-02, 0.23962965E-02,
         0.21306550E-02, 0.17725174E-02, 0.13268656E-02, 0.80657672E-03,
         0.23337584E-03, -0.36291778E-03, -0.94443569E-03, -0.14678875E-02,
         -0.18873741E-02, -0.21588493E-02, -0.22448371E-02, -0.21198179E-02,
         -0.17754602E-02, -0.12246034E-02, -0.50414167E-03, 0.32507078E-03,
         0.11811065E-02, 0.19673402E-02, 0.25827094E-02, 0.29342526E-02,
         0.29517083E-02, 0.26012166E-02, 0.18959062E-02, 0.90128649E-03,
         -0.26757144E-03, -0.14567885E-02, -0.24928550E-02, -0.32079776E-02,
         -0.34678637E-02, -0.31988217E-02, -0.24080969E-02, -0.11936240E-02,
         0.26134145E-03, 0.17117758E-02, 0.28906898E-02, 0.35614435E-02,
         0.35711778E-02, 0.28921374E-02, 0.16385898E-02, 0.49857464E-04,
         -0.15572671E-02, -0.28578151E-02, -0.35924212E-02, -0.36253044E-02,
         -0.29750860E-02, -0.18044436E-02, -0.37796664E-03, 0.10076215E-02,
         0.20937327E-02, 0.27090854E-02, 0.28031667E-02, 0.24276576E-02,
         0.17175597E-02, 0.81030795E-03, -0.12592304E-03, -0.94888491E-03,
         -0.15544816E-02, -0.18831972E-02, -0.19203142E-02, -0.16905849E-02,
         -0.12487737E-02, -0.66789911E-03, -0.27079461E-04, 0.59931935E-03,
         0.11499748E-02, 0.15816521E-02, 0.18709224E-02, 0.20129966E-02,
         0.20184702E-02, 0.19089181E-02, 0.17122289E-02, 0.14583770E-02,
         0.11760717E-02, 0.89046768E-03, 0.62190822E-03, 0.38553762E-03,
         0.19155022E-03, 0.45837109E-04, -0.49177834E-04, -0.93670762E-04,
         -0.89622968E-04, -0.401538532E-04],
        dtype=dtype)


def mcdint(et, dtype=tf.float32):
    """Multi Color Disk integration?

    Args:
        et: Energy Target?

    Return:
        value: UNKNOWN
    """

    NRES = 98            # Number of results? What is results?
    VALUE0 = 0.19321556  # UNKNOWN
    ET0 = 0.01           # Energy Target 0?
    STEP = 0.06          # UNKNOWN
    A = 0.52876731       # UNKNOWN
    B = 0.16637530       # UNKNOWN
    BEKI = -2.0 / 3.0    # BEKI means power in japanese

    # gc: Gauss UNKNOWN
    # gw: Gauss Width?
    # gn: Gauss UNKNOWN
    gc = tf.constant([0.78196667, -1.0662020, 1.1924180], dtype=dtype)
    gw = tf.constant([0.52078740, 0.51345700, 0.40779830], dtype=dtype)
    gn = tf.constant([0.37286910, 0.039775528, 0.037766505], dtype=dtype)

    res = precomputation(dtype=dtype)

    loget = tf.math.log(et) / tf.math.log(10.0)
    pos = (loget - tf.math.log(ET0)/tf.math.log(10.0)) / STEP + 1
    j = tf.cast(tf.math.floor(pos), tf.int32)
    pos = pos - tf.cast(j, dtype=dtype)
    pos = tf.clip_by_value(pos, 0.0, 1.0)
    resfact = tf.where(
        j < 1,
        res[0],
        tf.where(j >= NRES,
                 res[-1],
                 res[j-1] * (1.0 - pos) + res[j] * pos)
    )

    gaufact = 1.0
    for j in range(3):
        z = (loget - gc[j]) / gw[j]
        gaufact += gn[j] * tf.math.exp(-z * z / 2.0)

    value = VALUE0 * (et / ET0) ** BEKI * (
        1.0 + A * et ** B) * tf.math.exp(-et) * gaufact * (1.0 + resfact)
    return value


def mcdspc(E, Tin, Rin2):
    """Multi-Color Disk SPeCtrum.

    Args:
        E: Energy at which flux is computed.
        Tin: Temparature of inner disk.
        Rin2: Square of inner radius.

    Return:
        Flux: Flux at the input Energy point.
    """
    normfact = 361.0

    if Tin == 0.0:
        Flux = 0.0
    else:
        et = E / Tin
        value = mcdint(et)
        Flux = value * Tin * Tin * Rin2 / normfact

    return Flux


def xsdskb(EAR, NE, PARAM, IDT, dtype=tf.float32):
    """Xray spectrum disk black body, a.k.a. xsdskb.

    Args:
        EAR: Energy array?
        NE: Number of Energies?
        PARAM: Parameter.
        IDT: ?
        dtype: Data type.

    Return:
        PHOTAR: Photon values array.
        PHOTER: Photon errors array.
    """
    # Gaussian quadrature points.
    GAUSS = tf.constant([0.236926885, 0.478628670, 0.568888888,
                         0.478628670, 0.236926885, -0.906179846,
                         -0.538469310, 0.0, 0.538469310,
                         0.906179846], dtype=dtype)

    NE = tf.constant(NE, dtype=tf.int32)
    IDT = tf.constant(IDT, dtype=tf.int32)

    # PHOTOER: photon errors (zeros for diskbb)
    PHOTER = tf.zeros(NE, dtype=dtype)

    TIN = tf.cast(PARAM[0], dtype=dtype)

    EAR_shifted = EAR[1:]
    EAR_prev = EAR[:-1]
    XN = (EAR_shifted - EAR_prev) / 2.0

    PHOTAR = tf.TensorArray(dtype, size=NE)
    for i in range(NE):
        XH = XN[i] + EAR_prev[i]
        photon = tf.constant(0.0, dtype=dtype)
        for j in range(5):
            E = tf.cast(XN[i] * GAUSS[j + 5] + XH, dtype=dtype)
            photon += GAUSS[j] * mcdspc(E, TIN, 1.0)
        PHOTAR = PHOTAR.write(i, photon * XN[i])

    PHOTAR = PHOTAR.stack()

    return PHOTAR, PHOTER


def main():
    NE = 3491                          # NE: Number of Energies?
    EAR = tf.linspace(0.1, 20., NE+1)  # EAR: Energy ARray?
    PARAM = [0.2]                      # PARAM[0] is Tin for diskbb
    IDT = 1                            # ???

    PHOTAR, PHOTER = xsdskb(EAR, NE, PARAM, IDT)
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
