#!/usr/bin/env python3
# ===================== Imports =====================
import numpy as np
import pickle, os
from pathlib import Path
from scipy.special import spherical_jn
import camb
from camb import model

# ===================== IO =====================
def atomic_pickle_dump(obj, filename):
    filename = str(filename)
    tmp = filename + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, filename)

def safe_pickle_load(filename):
    if not os.path.exists(filename):
        return {}
    with open(filename, "rb") as f:
        return pickle.load(f)

# ===================== SETTINGS =====================
Lmax = 10

# external discrete radii: r0, r1', r2', r3, r3'
r_choices = np.array([0.0, 50.0, 100.0], dtype=np.float64)
Nr = len(r_choices)

# dense grids for s and r
smin, smax = 0.0, 200.0
Ns = 160
s_grid = np.linspace(smin, smax, Ns, dtype=np.float64)
r_grid = np.linspace(smin, smax, Ns, dtype=np.float64)

kmin, kmax = 1e-4, 3.0
Nk = 240
k = np.logspace(np.log10(kmin), np.log10(kmax), Nk)
dlnk = np.log(k[1] / k[0])
tpi = 1.0 / (2.0 * np.pi**2)

H_PKL = Path("h_tables.pkl")

# ===================== LOAD EXISTING =====================
Htab = safe_pickle_load(H_PKL)
print(f"Loaded {len(Htab)} existing h entries", flush=True)

# ===================== CAMB =====================
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
pars.set_matter_power(kmax=kmax)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
_, _, Plin_k = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=Nk)
Pk = Plin_k[0]

# ===================== BESSELS =====================
# discrete external-radius Bessels
j_ext = np.zeros((Lmax + 1, Nr, Nk), dtype=np.float64)

# dense s-grid Bessels
j_s = np.zeros((Lmax + 1, Ns, Nk), dtype=np.float64)

# dense r-grid Bessels
j_r = np.zeros((Lmax + 1, Ns, Nk), dtype=np.float64)

for L in range(Lmax + 1):
    for i, rv in enumerate(r_choices):
        j_ext[L, i, :] = spherical_jn(L, k * rv)

    j_s[L, :, :] = spherical_jn(L, k[None, :] * s_grid[:, None])
    j_r[L, :, :] = spherical_jn(L, k[None, :] * r_grid[:, None])

# ===================== WEIGHT =====================
# n+n'_2 = 0 and n+n'_3 = 0  => k^(0+3)
w0 = dlnk * tpi * Pk * k**3

# ===================== h TABLE =====================
# h_{L1,L2,L3,L4}(r0, s, r1', r)
# shape = (Nr, Ns, Nr, Ns)
def compute_hA(L1, L2, L3, L4):
    return np.einsum(
        "ik,ak,jk,bk,k->iajb",
        j_ext[L1],   # r0   -> Nr
        j_s[L2],     # s    -> Ns
        j_ext[L3],   # r1'  -> Nr
        j_r[L4],     # r    -> Ns
        w0,
        optimize=True
    ).astype(np.float32)

for L1 in range(Lmax + 1):
    for L2 in range(Lmax + 1):
        for L3 in range(Lmax + 1):
            for L4 in range(Lmax + 1):
                key = ("hA", L1, L2, L3, L4)

                if key in Htab:
                    continue

                Htab[key] = compute_hA(L1, L2, L3, L4)

        atomic_pickle_dump(Htab, H_PKL)
        print(f"checkpoint h: L1={L1}, L2={L2}", flush=True)

print("DONE building h tables", flush=True)