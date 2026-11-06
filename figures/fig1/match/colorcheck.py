import numpy as np, itertools

PROT = np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]])
DEUT = np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]])


def hex2lin(h):
    c = np.array([int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2hex(l):
    l = np.clip(l, 0, 1)
    c = np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)
    return "#" + "".join(f"{int(round(x * 255)):02x}" for x in c)


M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)


def lin2oklab(l):
    return M2 @ np.cbrt(M1 @ l)


def oklab2lin(lab):
    M2i = np.linalg.inv(M2)
    M1i = np.linalg.inv(M1)
    return M1i @ ((M2i @ lab) ** 3)


def dE(a, b, sim=None):
    la, lb = hex2lin(a), hex2lin(b)
    if sim is not None:
        la, lb = np.clip(sim @ la, 0, 1), np.clip(sim @ lb, 0, 1)
    return 100 * np.linalg.norm(lin2oklab(la) - lin2oklab(lb))


def report(cols, pairs=None):
    pairs = pairs or list(itertools.combinations(range(len(cols)), 2))
    out = {}
    for nm, sim in (("normal", None), ("prot", PROT), ("deut", DEUT)):
        v = [(dE(cols[i], cols[j], sim), cols[i], cols[j]) for i, j in pairs]
        out[nm] = min(v)
    return out


def L(h):
    return lin2oklab(hex2lin(h))[0]
