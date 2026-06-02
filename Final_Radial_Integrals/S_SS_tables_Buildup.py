#!/usr/bin/env python3
import numpy as np
import pickle, os
from pathlib import Path

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

# discrete external radii
# index 0 -> r=0
# index 1 -> r=50
# index 2 -> r=100
r_choices = np.array([0.0, 50.0, 100.0], dtype=np.float64)
Nr = len(r_choices)

# dense grids for s and r integrations
smin, smax = 0.0, 200.0
Ns = 160
s_grid = np.linspace(smin, smax, Ns, dtype=np.float64)
r_grid = np.linspace(smin, smax, Ns, dtype=np.float64)

ds = s_grid[1] - s_grid[0]
dr = r_grid[1] - r_grid[0]

# n_23 = 2 => r^(n_23 - 1) = r^1
r_weight = r_grid
s2 = s_grid**2

G_PKL = Path("/blue/zslepian/wortola/4PCF_Cov/4PCF_Cov_Precomputed_Constants_and_Radial_Int/g_table_precomputation/g_tables_P_S_n0.pkl")
H_PKL = Path("/blue/zslepian/wortola/4PCF_Cov/4PCF_Cov_Precomputed_Constants_and_Radial_Int/h_table_precomputation/h_tables.pkl")
S_PKL = Path("S_SS_tables.pkl")

# ===================== LOAD TABLES =====================
Gtab = safe_pickle_load(G_PKL)
Htab = safe_pickle_load(H_PKL)
Stab = safe_pickle_load(S_PKL)

print(f"Loaded {len(Gtab)} g entries", flush=True)
print(f"Loaded {len(Htab)} h entries", flush=True)
print(f"Loaded {len(Stab)} existing S_SS entries", flush=True)

# ===================== FIXED CONSTRAINTS =====================
# Lq20 = Lq30 = Lq10 = L'q10 = 0
Lq10 = 0
Lq10p = 0
Lq20 = 0
Lq30 = 0

# Lq1s = 0
Lq1s = 0

# Lq2s = L'q21
# Lq3s = L'q32

# r0 = r0' = 0  -> use index 0
IR0 = 0

# we only want final external radii 50,100 -> indices 1,2
radial_eval_indices = [1, 2]

# ===================== HELPERS =====================
def trapz_s(y):
    # y shape (Ns,)
    return ds * (0.5 * (y[0] + y[-1]) + np.sum(y[1:-1]))

def trapz_r(y):
    # y shape (Ns,)
    return dr * (0.5 * (y[0] + y[-1]) + np.sum(y[1:-1]))

# ===================== MAIN LOOPS =====================
# key = (L'q21, L'q32, ell_2, L33, L3s, L'33)
for Lq21p in range(Lmax + 1):
    for Lq32p in range(Lmax + 1):
        for ell2 in range(Lmax + 1):
            for L33 in range(Lmax + 1):
                for L3s in range(Lmax + 1):
                    for L33p in range(Lmax + 1):

                        key = (Lq21p, Lq32p, ell2, L33, L3s, L33p)

                        if key in Stab:
                            continue

                        # ---------- g_{Lq10, L'q1s, L'q10} = g_{0,0,0}(r0,s,r0')
                        key_g1 = ("gA", Lq10, Lq1s, Lq10p)
                        if key_g1 not in Gtab:
                            continue
                        g1 = Gtab[key_g1][IR0, IR0, :]   # shape (Ns,)

                        # ---------- h_{Lq20,Lq2s,L'q21,ell2} = h_{0,L'q21,L'q21,ell2}(r0,s,r1',r)
                        key_h1 = ("hA", Lq20, Lq21p, Lq21p, ell2)
                        if key_h1 not in Htab:
                            continue
                        h1 = Htab[key_h1][IR0, :, :, :]  # shape (Ns, Nr, Ns)

                        # ---------- h_{Lq30,Lq3s,L'q32,ell2} = h_{0,L'q32,L'q32,ell2}(r0,s,r2',r)
                        key_h2 = ("hA", Lq30, Lq32p, Lq32p, ell2)
                        if key_h2 not in Htab:
                            continue
                        h2 = Htab[key_h2][IR0, :, :, :]  # shape (Ns, Nr, Ns)

                        # ---------- g_{L33,L3s,L'33}(r3,s,r3')
                        key_g2 = ("gA", L33, L3s, L33p)
                        if key_g2 not in Gtab:
                            continue
                        g2 = Gtab[key_g2]  # shape (Nr, Nr, Ns)

                        # store only r = 50,100 outputs as a length-2 array
                        Svals = np.zeros(len(radial_eval_indices), dtype=np.float32)

                        for out_i, ir in enumerate(radial_eval_indices):
                            # use same external radius choice for r1', r2', r3, r3'
                            # r1' index = ir
                            # r2' index = ir
                            # r3  index = ir
                            # r3' index = ir

                            # h1(s,r) at fixed r1'
                            h1_fixed = h1[:, ir, :]   # shape (Ns, Ns)

                            # h2(s,r) at fixed r2'
                            h2_fixed = h2[:, ir, :]   # shape (Ns, Ns)

                            # g2(s) at fixed r3, r3'
                            g2_fixed = g2[ir, ir, :]  # shape (Ns,)

                            # integrand over s and r:
                            # ∫ dr r^(n23-1) ∫ ds s^2 g1(s) h1(s,r) h2(s,r) g2(s)
                            # first integrate over s for each r
                            integrand_r = np.zeros(Ns, dtype=np.float64)

                            for ir_dense in range(Ns):
                                integrand_s = s2 * g1 * h1_fixed[:, ir_dense] * h2_fixed[:, ir_dense] * g2_fixed
                                integrand_r[ir_dense] = trapz_s(integrand_s)

                            integrand_r *= r_weight
                            Svals[out_i] = trapz_r(integrand_r)

                        Stab[key] = Svals.astype(np.float32)

                atomic_pickle_dump(Stab, S_PKL)
                print(f"checkpoint S_SS: Lq21p={Lq21p}, Lq32p={Lq32p}, ell2={ell2}, L33={L33}", flush=True)

print("DONE building S_SS tables", flush=True)