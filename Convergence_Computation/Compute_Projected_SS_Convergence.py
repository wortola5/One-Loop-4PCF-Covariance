#!/usr/bin/env python3
import os
import pickle
import numpy as np
from pathlib import Path
from functools import lru_cache
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================
BASE_PRE = Path("/blue/zslepian/wortola/4PCF_Cov/4PCF_Cov_Precomputed_Constants_and_Radial_Int")
BASE_OUT = Path("/blue/zslepian/wortola/4PCF_Cov")

Q_DIR = BASE_PRE / "Q_SS_Computtation" / "Qbar_SS_blocks"
J_DIR = BASE_PRE / "Precompute_J_with_m" / "Modified_J_m_blocks"
L_DIR = BASE_PRE / "Precompute_L_Tensor" / "L_tensor_blocks"

C_PATH = BASE_PRE / "Cprime_Gamma_all.pkl"
G_PATH = BASE_PRE / "Modified_G_all.pkl"
S_PATH = BASE_PRE / "S_SS_table_precomputation" / "S_SS_tables.pkl"

F_PATH = Path("/orange/zslepian/ortolaw/4PCF_Theory_Code_Up/4PCF_theory_code_up/Radial_Integrals/T311/f_integrals.npz")

OUTDIR = BASE_OUT / "Projected_SS_Convergence_2"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# IO
# ============================================================
def load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def atomic_pickle_dump(obj, filename):
    filename = Path(filename)
    tmp = filename.parent / f"{filename.name}.tmp.{os.getpid()}"
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, filename)

def get(d, key, default=0.0):
    return d.get(key, default)

# ============================================================
# LOAD STATIC INGREDIENTS
# ============================================================
Cprime = load_pkl(C_PATH)
Gmod   = load_pkl(G_PATH)
S_tab  = load_pkl(S_PATH)

f_all = np.load(F_PATH, allow_pickle=True)["f"].item()

# ============================================================
# SETTINGS
# ============================================================
LREF = 10
jmax = 2

r_labels = ["50", "100"]
colors = ["blue", "red"]

# four nonzero projected cases:
# (L11,L33,L'q21,L'q32,L'33)
EXTERNAL_CASES = [
    ((0,0,0,0,0), "ext_000_000"),
    ((1,0,1,1,0), "ext_110_110"),
    ((1,0,1,0,1), "ext_110_101"),
    ((1,0,0,1,1), "ext_110_011"),
]

PREFAC = 1.0

# ============================================================
# RADIUS INDICES FOR f-INTEGRALS
# ============================================================
# These reproduce the grids from the f-integral construction
rmin = 0.0
rmax = 200.0
numr = 200
numri = 80

r_grid_f = np.linspace(rmin, rmax, numr) + 5.0
ri_grid_f = np.linspace(rmin, rmax, numri) + 5.0

def nearest_index(arr, value):
    return int(np.argmin(np.abs(arr - value)))

# Use the closest stored values to 50 and 100
f_r_indices  = [nearest_index(r_grid_f, 50.0),  nearest_index(r_grid_f, 100.0)]
f_ri_indices = [nearest_index(ri_grid_f, 50.0), nearest_index(ri_grid_f, 100.0)]

# ============================================================
# BLOCK LOADERS
# ============================================================
@lru_cache(maxsize=None)
def load_Q_block(L33: int, L3sp: int, L33p: int):
    path = Q_DIR / f"Qbar_SS_L33_{int(L33):02d}_L3sp_{int(L3sp):02d}_L33p_{int(L33p):02d}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=None)
def load_J_block(L1: int, L2: int, L3: int):
    path = J_DIR / f"Modified_J_m_included_L1_{int(L1):02d}_L2_{int(L2):02d}_L3_{int(L3):02d}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=None)
def load_L_block(L1: int, L2: int, L3: int):
    path = L_DIR / f"L_tensor_L1_{int(L1):02d}_L2_{int(L2):02d}_L3_{int(L3):02d}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)

# ============================================================
# PROJECTED S-S COVARIANCE FOR ONE EXTERNAL CASE
# ============================================================
def compute_projected_SS_case(ext_case, Lref=LREF):
    """
    External labels:
      (L11, L33 ; L'q21, L'q32, L'33)
    corresponding to
      Lambda  = {0, L11, L11, L33}
      Lambda' = {0, L'q21, L'q32, L'33}

    Internal labels truncated by Lcut:
      L'3s, j12, j13, j23, j, ell2

    with:
      j12,j13,j23,j only ranging over 0..2,
      and L'3s, ell2 ranging over 0..Lcut.
    """

    L11, L33, Lq21p, Lq32p, L33p = ext_case

    # raw shell contributions, indexed by shell = max(internal labels)
    shell_contrib = np.zeros((Lref + 1, 2), dtype=np.complex128)

    hits = {L: 0 for L in range(Lref + 1)}
    missing = {
        L: {
            "Q_block": set(), "Q": set(),
            "Cpq1": set(), "Cpq2": set(), "Cpq3": set(), "Cp33": set(),
            "G33": set(), "S": set(), "J": set(), "L1": set(), "L2": set(),
            "f": set()
        }
        for L in range(Lref + 1)
    }

    # ---- fixed factors for this external case ----
    # C_{Lq10,Lq1s,L'q10} Gamma_{...} with reductions: (0,0,0)
    key_Cq1 = (0, 0, 0)
    Cq1 = get(Cprime, key_Cq1, 0.0)
    if key_Cq1 not in Cprime:
        for L in range(Lref + 1):
            missing[L]["Cpq1"].add(key_Cq1)

    key_Cq2 = (0, Lq21p, Lq21p)
    Cq2 = get(Cprime, key_Cq2, 0.0)
    if key_Cq2 not in Cprime:
        for L in range(Lref + 1):
            missing[L]["Cpq2"].add(key_Cq2)

    key_Cq3 = (0, Lq32p, Lq32p)
    Cq3 = get(Cprime, key_Cq3, 0.0)
    if key_Cq3 not in Cprime:
        for L in range(Lref + 1):
            missing[L]["Cpq3"].add(key_Cq3)

    if Cq1 == 0.0 or Cq2 == 0.0 or Cq3 == 0.0:
        Cproj = np.zeros((Lref + 1, 2), dtype=np.float64)
        return Cproj, hits, missing

    # f^{[0]}_{L11,L11}(r1,r2), using nearest stored values to 50 and 100
    try:
        f_table = f_all[0][L11][L11]
        f_vec = np.array([
            f_table[f_r_indices[0],  f_ri_indices[0]],
            f_table[f_r_indices[1],  f_ri_indices[1]]
        ], dtype=np.complex128)
    except Exception:
        for L in range(Lref + 1):
            missing[L]["f"].add((0, L11, L11))
        Cproj = np.zeros((Lref + 1, 2), dtype=np.float64)
        return Cproj, hits, missing

    # J-block fixed by reductions Lq10=Lq1s=L'q10=0
    try:
        J000 = load_J_block(0, 0, 0)
    except FileNotFoundError:
        for L in range(Lref + 1):
            missing[L]["J"].add(("block", 0, 0, 0))
        Cproj = np.zeros((Lref + 1, 2), dtype=np.float64)
        return Cproj, hits, missing

    # L-tensor blocks fixed by external primed labels:
    # first  L : (0, L'q21, L'q21)
    # second L : (0, L'q32, L'q32)
    try:
        Lblock_q21 = load_L_block(0, Lq21p, Lq21p)
    except FileNotFoundError:
        for L in range(Lref + 1):
            missing[L]["L1"].add(("block", 0, Lq21p, Lq21p))
        Cproj = np.zeros((Lref + 1, 2), dtype=np.float64)
        return Cproj, hits, missing

    try:
        Lblock_q32 = load_L_block(0, Lq32p, Lq32p)
    except FileNotFoundError:
        for L in range(Lref + 1):
            missing[L]["L2"].add(("block", 0, Lq32p, Lq32p))
        Cproj = np.zeros((Lref + 1, 2), dtype=np.float64)
        return Cproj, hits, missing

    # ---- internal sums ----
    for L3sp in range(0, Lref + 1):

        # factors that depend on internal L'3s
        key_C33 = (L33, L33p, L3sp)
        C33 = get(Cprime, key_C33, 0.0)
        if key_C33 not in Cprime:
            for L in range(L3sp, Lref + 1):
                missing[L]["Cp33"].add(key_C33)

        G33 = get(Gmod, key_C33, 0.0)
        if key_C33 not in Gmod:
            for L in range(L3sp, Lref + 1):
                missing[L]["G33"].add(key_C33)

        if C33 == 0.0 or G33 == 0.0:
            continue

        # Q block keyed by (L11,L'q21,L33,L'3s,L'33,L'q32)
        try:
            Q_block = load_Q_block(L33, L3sp, L33p)
        except FileNotFoundError:
            for L in range(L3sp, Lref + 1):
                missing[L]["Q_block"].add((L33, L3sp, L33p))
            continue

        keyQ = (L11, Lq21p, L33, L3sp, L33p, Lq32p)
        Qval = get(Q_block, keyQ, 0.0)
        if keyQ not in Q_block:
            for L in range(L3sp, Lref + 1):
                missing[L]["Q"].add(keyQ)

        if Qval == 0.0:
            continue

        for ell2 in range(0, Lref + 1):

            keyS = (Lq21p, Lq32p, ell2, L33, L3sp, L33p)
            Sval = get(S_tab, keyS, None)
            if Sval is None:
                for L in range(max(L3sp, ell2), Lref + 1):
                    missing[L]["S"].add(keyS)
                continue

            Svec = np.asarray(Sval, dtype=np.complex128)
            if np.all(Svec == 0.0):
                continue

            phase = (-1)**int(ell2)

            for j12 in range(0, jmax + 1):
                for j13 in range(0, jmax + 1):
                    for j23 in range(0, jmax + 1):
                        for j in range(0, jmax + 1):

                            shell = max(L3sp, ell2, j12, j13, j23, j)
                            if shell > Lref:
                                continue

                            for mj12 in range(-j12, j12 + 1):
                                for mj13 in range(-j13, j13 + 1):

                                    keyJ = (0, 0, 0, j12, j13, mj12, mj13)
                                    Jval = get(J000, keyJ, 0.0)
                                    if keyJ not in J000:
                                        missing[shell]["J"].add(keyJ)
                                    if Jval == 0.0:
                                        continue

                                    for mj23 in range(-j23, j23 + 1):
                                        for mj in range(-j, j + 1):
                                            for m2 in range(-ell2, ell2 + 1):

                                                keyL1 = (j12, j23, j, ell2,
                                                         0, Lq21p, Lq21p,
                                                         mj12, mj23, mj, m2)
                                                L1val = get(Lblock_q21, keyL1, 0.0)
                                                if keyL1 not in Lblock_q21:
                                                    missing[shell]["L1"].add(keyL1)
                                                if L1val == 0.0:
                                                    continue

                                                keyL2 = (j13, j23, j, ell2,
                                                         0, Lq32p, Lq32p,
                                                         mj13, mj23, mj, m2)
                                                L2val = get(Lblock_q32, keyL2, 0.0)
                                                if keyL2 not in Lblock_q32:
                                                    missing[shell]["L2"].add(keyL2)
                                                if L2val == 0.0:
                                                    continue

                                                contrib = (
                                                    phase
                                                    * Qval
                                                    * Cq1 * Cq2 * Cq3 * C33
                                                    * Jval
                                                    * L1val * L2val
                                                    * G33
                                                )

                                                shell_contrib[shell, :] += contrib * f_vec * Svec
                                                hits[shell] += 1

    Cproj = PREFAC * np.cumsum(shell_contrib, axis=0)
    Cproj = np.real_if_close(Cproj, tol=1000)

    if np.iscomplexobj(Cproj):
        Cproj = Cproj.real

    return Cproj.astype(np.float64), hits, missing

# ============================================================
# PLOTS
# ============================================================
def make_single_plot(ext_case, label, R):
    L11, L33, Lq21p, Lq32p, L33p = ext_case

    fig, ax = plt.subplots(figsize=(6, 4))
    Lvals = np.arange(R.shape[0])

    for i, rlab in enumerate(r_labels):
        ax.plot(
            Lvals, R[:, i],
            marker='o',
            color=colors[i],
            label=rf"$r \;=\; {rlab} \; [\mathrm{{Mpc}}/h]$"
        )

    ax.set_title(
        rf"$\Lambda = \{{0,{L11},{L11},{L33}\}},\ \Lambda' = \{{0,{Lq21p},{Lq32p},{L33p}\}}$",
        fontsize=16
    )
    ax.set_xlabel(r"$L_{\rm Cutoff}$", fontsize=16)
    ax.set_ylabel(r"$R(L_{\rm Cutoff})$", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18)

    fig.tight_layout()
    fig.savefig(OUTDIR / f"{label}_FINAL_plot.pdf")
    plt.close(fig)

def make_stacked_plot(all_R, case_map):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax in axes:
        ax.set_visible(False)

    fig.suptitle(r"Projected SS Covariance Convergence", fontsize=18)

    for idx, label in enumerate(case_map):
        ext_case = case_map[label]
        R = all_R[label]

        L11, L33, Lq21p, Lq32p, L33p = ext_case

        ax = axes[idx]
        ax.set_visible(True)

        Lvals = np.arange(R.shape[0])

        for i, rlab in enumerate(r_labels):
            ax.plot(
                Lvals, R[:, i],
                marker='o',
                color=colors[i],
                label=rf"$r \;=\; {rlab} \; [\mathrm{{Mpc}}/h]$"
            )

        ax.set_title(
            rf"$\Lambda = \{{0,{L11},{L11},{L33}\}},\ \Lambda' = \{{0,{Lq21p},{Lq32p},{L33p}\}}$",
            fontsize=12
        )
        ax.set_xlabel(r"$L_{\rm Cutoff}$", fontsize=14)
        ax.set_ylabel(r"$R(L_{\rm Cutoff})$", fontsize=14)
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=18)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / "STACKED_FINAL_SS_convergence.pdf")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main():
    all_R = {}
    case_map = {label: ext_case for ext_case, label in EXTERNAL_CASES}

    for ext_case, label in EXTERNAL_CASES:
        print(f"Computing {label} ...", flush=True)

        Cproj, hits, missing = compute_projected_SS_case(ext_case, Lref=LREF)

        np.save(OUTDIR / f"{label}_Cproj.npy", Cproj)

        R = np.zeros_like(Cproj)
        ref = Cproj[LREF, :]

        for i in range(Cproj.shape[1]):
            if ref[i] != 0.0:
                R[:, i] = Cproj[:, i] / ref[i]
            else:
                R[:, i] = 0.0

        np.save(OUTDIR / f"{label}_R.npy", R)
        all_R[label] = R

        total_missing = 0
        for L in range(LREF + 1):
            for k in missing[L]:
                total_missing += len(missing[L][k])

        if total_missing > 0:
            atomic_pickle_dump(
                {"hits": hits, "missing": missing},
                OUTDIR / f"{label}_diag.pkl"
            )

        make_single_plot(ext_case, label, R)
        print(f"Done {label}", flush=True)

    make_stacked_plot(all_R, case_map)
    print("All projected S-S convergence outputs saved.", flush=True)

if __name__ == "__main__":
    main()