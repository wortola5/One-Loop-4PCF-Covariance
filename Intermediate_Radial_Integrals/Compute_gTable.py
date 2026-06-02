
# =================== Imports ===================

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
Lmax = 10 #change to desired Lmax
r_choices = np.array([0.0, 50.0, 100.0, 150.0])
Nr = len(r_choices)

smin, smax = 0.0, 200.0
Ns = 160
s_grid = np.linspace(smin, smax, Ns)

kmin, kmax = 1e-4, 3.0
Nk = 240
k = np.logspace(np.log10(kmin), np.log10(kmax), Nk)
dlnk = np.log(k[1]/k[0])
tpi = 1.0 / (2.0*np.pi**2)

G_PKL = Path("g_tables.pkl")

# ===================== LOAD EXISTING =====================
Gtab = safe_pickle_load(G_PKL)
print(f"Loaded {len(Gtab)} existing g entries")

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
j_r = np.zeros((Lmax+1, Nr, Nk)) 
j_s = np.zeros((Lmax+1, Ns, Nk))

for L in range(Lmax+1):
    for i, rv in enumerate(r_choices):
        j_r[L, i, :] = spherical_jn(L, k*rv)
    j_s[L, :, :] = spherical_jn(L, k[None, :]*s_grid[:, None])

# ===================== WEIGHT =====================
w0 = dlnk * tpi * Pk * k**3

# ===================== gA =====================
def compute_gA(L1, L1p, L1s):
    return np.einsum(
        "ik,jk,ak,k->ija",
        j_r[L1], #Choose L, integrate over k -> Size #(Nr)
        j_r[L1p],
        j_s[L1s],#Choose L, integrate over k -> Size #(Ns)
        w0,
        optimize=True
    ).astype(np.float32) #Final result -> Size #(Nr,Nr,Ns)

#Gtab = {}

for L1 in range(Lmax+1):
    for L1p in range(Lmax+1):
        for L1s in range(Lmax+1):
            
            key = ("gA", L1, L1p, L1s)

            # Skip if already computed
            if key in Gtab:
                continue
            
            Gtab[key] = compute_gA(L1, L1p, L1s)
            
            #Gtab[("gA", L1, L1p, L1s)] = compute_gA(L1, L1p, L1s)
    atomic_pickle_dump(Gtab, G_PKL)
    print(f"checkpoint gA: L1={L1}", flush=True)

print("DONE building gA tables")






