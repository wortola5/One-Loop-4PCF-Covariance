from functools import lru_cache
from pathlib import Path
import argparse
import pickle
import numpy as np
from sympy.physics.wigner import wigner_3j


# ============================================================
# Basic angular objects 
# ============================================================

@lru_cache(maxsize=None)
def W3j(l1, l2, l3, m1, m2, m3):
    return float(wigner_3j(int(l1), int(l2), int(l3), int(m1), int(m2), int(m3)))


@lru_cache(maxsize=None)
def Gaunt(l1, l2, l3, m1, m2, m3):
    """
    Gaunt(l1,l2,l3; m1,m2,m3) =
      sqrt((2l1+1)(2l2+1)(2l3+1)/(4π)) *
      (l1 l2 l3; m1 m2 m3)(l1 l2 l3; 0 0 0)
    """
    pref = np.sqrt((2*int(l1)+1)*(2*int(l2)+1)*(2*int(l3)+1)/(4*np.pi))
    return pref * W3j(l1, l2, l3, m1, m2, m3) * W3j(l1, l2, l3, 0, 0, 0)


@lru_cache(maxsize=None)
def fancyC(l1, l2, l3, m1, m2, m3):
    """
    Same 3-harmonic coupling convention as in the notebook.
    """
    return (-1)**(int(l1) + int(l2) + int(l3)) * W3j(l1, l2, l3, m1, m2, m3)


@lru_cache(maxsize=None)
def D_of_L(L):
    """
    D_L convention used elsewhere in the covariance code.
    """
    return ((-1)**int(L)) / np.sqrt(2*int(L) + 1)


# ============================================================
# Qbar_SS 
# ============================================================

@lru_cache(maxsize=None)
def Qbar_SS_special(L11, Lq21p, L33, L3sp, L33p, Lq32p):
    r"""
    Precompute the angular constant Qbar_SS in the special case r0 = r0' = 0.

    Constraints applied directly:
      Lq20 = 0 = Lq30 = Lq10 = L'q10
      Lq1s = 0
      Lq2s = L'q21
      Lq3s = L'q32

      Mq20 = 0 = Mq30 = Mq10 = M'q10
      Mq1s = 0
      Mq2s = -M'q21
      Mq3s = -M'q32

    The 4-harmonic coupling constants reduce to 3-harmonic coupling constants:
      C^Lambda_M   -> fancyC(L11, L11, L33,  M11, -M11, M33)
      C^Lambda'_M' -> fancyC(Lq21p, Lq32p, L33p, Mq21p, Mq32p, M33p)

    We keep the remaining sums explicit and let the selection rules in the
    Wigner/Gaunt objects enforce the surviving terms.

    Final key:
      (L11, L'q21, L33, L'3s, L'33, L'q32)
    """

    # Immediate triangle/parity pruning from the reduced 3-harmonic couplings
    # (not required for correctness, just avoids useless work)
    if abs(int(Lq21p) - int(Lq32p)) > int(L33p):
        return 0.0 + 0.0j
    if int(L33p) > int(Lq21p) + int(Lq32p):
        return 0.0 + 0.0j
    if (int(Lq21p) + int(Lq32p) + int(L33p)) % 2 != 0:
        return 0.0 + 0.0j

    if abs(int(L11) - int(L11)) > int(L33):
        return 0.0 + 0.0j
    if int(L33) > int(L11) + int(L11):
        return 0.0 + 0.0j
    if (2*int(L11) + int(L33)) % 2 != 0:
        return 0.0 + 0.0j

    res = 0.0 + 0.0j

    # In Eq. (C.8) after the special-case constraints, the global phase is
    # (-1)^(M11 + L33), together with D_{L11}.
    DL11 = D_of_L(L11)

    # The C^{0,0,0}_{0,0,0} factor from the Lq10/Lq1s/L'q10 piece.
    C000 = fancyC(0, 0, 0, 0, 0, 0)

    # Independent magnetic sums we keep explicit.
    for M11 in range(-int(L11), int(L11) + 1):
        for Mq21p in range(-int(Lq21p), int(Lq21p) + 1):
            for Mq32p in range(-int(Lq32p), int(Lq32p) + 1):
                for M3sp in range(-int(L3sp), int(L3sp) + 1):
                    for M33 in range(-int(L33), int(L33) + 1):
                        for M33p in range(-int(L33p), int(L33p) + 1):

                            # Constraints already extracted from the special case
                            Mq2s = -Mq21p
                            Mq3s = -Mq32p
                            Mq20 = 0
                            Mq30 = 0
                            Mq10 = 0
                            Mq1s = 0
                            Mq10p = 0

                            # Reduced 3-harmonic couplings from the R and R' averages
                            C_Lambda_p = fancyC(Lq21p, Lq32p, L33p, Mq21p, Mq32p, M33p)
                            if C_Lambda_p == 0:
                                continue

                            C_Lambda = fancyC(L11, L11, L33, M11, -M11, M33)
                            if C_Lambda == 0:
                                continue

                            # The remaining C-factors from Eq. (C.8) after constraints
                            C_q2 = fancyC(0, Lq21p, Lq21p, 0, Mq2s, Mq21p)
                            if C_q2 == 0:
                                continue

                            C_mid = fancyC(L33, L3sp, L33p, M33, M3sp, M33p)
                            if C_mid == 0:
                                continue

                            C_q3 = fancyC(0, Lq32p, Lq32p, 0, Mq3s, Mq32p)
                            if C_q3 == 0:
                                continue

                            phase = (-1)**(int(M11) + int(L33))
                            pref = phase * DL11 * C_q2 * C_mid * C_q3 * C000 * C_Lambda_p * C_Lambda

                            # Bottom Gaunt chain in Eq. (C.8) with the special-case substitutions:
                            #   G^{Lq2s,L'3s,ell}_{Mq2s,M'3s,-m}
                            #   G^{ell,Lq3s,Lq1s}_{m,Mq3s,Mq1s}
                            #   G^{0,0,ell'}_{0,0,-m'}
                            #   G^{ell',0,L33}_{m',0,-M33}
                            ell_min = abs(int(Lq21p) - int(L3sp))
                            ell_max = int(Lq21p) + int(L3sp)

                            for ell in range(ell_min, ell_max + 1):
                                # From the second Gaunt, since Lq1s = 0 and Mq1s = 0,
                                # m is fixed by m + Mq3s + 0 = 0.
                                m = -Mq3s
                                if abs(m) > ell:
                                    continue

                                G1 = Gaunt(Lq21p, L3sp, ell, Mq2s, M3sp, -m)
                                if G1 == 0:
                                    continue

                                G2 = Gaunt(ell, Lq32p, 0, m, Mq3s, 0)
                                if G2 == 0:
                                    continue

                                # Keep ell',m' explicit in the same spirit as the notebook,
                                # while the Gaunt selection rules collapse the sum.
                                for ellp in range(0, int(L33) + 1):
                                    mp = 0
                                    if abs(mp) > ellp:
                                        continue

                                    G3 = Gaunt(0, 0, ellp, 0, 0, -mp)
                                    if G3 == 0:
                                        continue

                                    G4 = Gaunt(ellp, 0, L33, mp, 0, -M33)
                                    if G4 == 0:
                                        continue

                                    res += pref * G1 * G2 * G3 * G4

    return complex(res)


# ============================================================
# Block computation: one block = fixed (L33, L'3s, L'33)
# ============================================================

def block_from_task_id(task_id, Lmax):
    n = int(Lmax) + 1
    total = n**3
    if task_id < 0 or task_id >= total:
        raise ValueError(f"task_id={task_id} is out of range for Lmax={Lmax}; expected 0..{total-1}")

    L33 = task_id // (n * n)
    rem = task_id % (n * n)
    L3sp = rem // n
    L33p = rem % n
    return int(L33), int(L3sp), int(L33p)


def compute_block(task_id, Lmax, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    L33, L3sp, L33p = block_from_task_id(task_id, Lmax)
    outfile = outdir / f"Qbar_SS_L33_{L33:02d}_L3sp_{L3sp:02d}_L33p_{L33p:02d}.pkl"
    tmpfile = outdir / f"Qbar_SS_L33_{L33:02d}_L3sp_{L3sp:02d}_L33p_{L33p:02d}.tmp"

    if outfile.exists():
        with outfile.open("rb") as f:
            data = pickle.load(f)
    else:
        data = {}

    for L11 in range(0, int(Lmax) + 1):
        for Lq21p in range(0, int(Lmax) + 1):
            for Lq32p in range(0, int(Lmax) + 1):
                key = (int(L11), int(Lq21p), int(L33), int(L3sp), int(L33p), int(Lq32p))
                if key in data:
                    continue
                data[key] = complex(Qbar_SS_special(L11, Lq21p, L33, L3sp, L33p, Lq32p))

        # incremental save per L11 so a long block can resume cleanly
        with tmpfile.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmpfile.replace(outfile)

    return outfile


# ============================================================
# Merge all blocks
# ============================================================

def merge_blocks(outdir, merged_name="Qbar_SS.pkl"):
    outdir = Path(outdir)
    files = sorted(outdir.glob("Qbar_SS_L33_*_L3sp_*_L33p_*.pkl"))

    merged = {}
    for fp in files:
        with fp.open("rb") as f:
            block = pickle.load(f)
        merged.update(block)

    merged_path = outdir / merged_name
    tmp_path = outdir / f"{merged_name}.tmp"
    with tmp_path.open("wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(merged_path)

    return merged_path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Precompute Qbar_SS in blocks of (L33, L'3s, L'33).")
    parser.add_argument("--Lmax", type=int, default=16)
    parser.add_argument("--task-id", type=int, default=None, help="Array task id selecting one (L33,L'3s,L'33) block.")
    parser.add_argument("--outdir", type=str, default="Qbar_SS_blocks")
    parser.add_argument("--merge", action="store_true", help="Merge all block pickles into Qbar_SS.pkl")
    args = parser.parse_args()

    if args.merge:
        merged = merge_blocks(args.outdir, merged_name="Qbar_SS.pkl")
        print(f"Merged blocks into: {merged}")
        return

    if args.task_id is None:
        raise ValueError("Provide --task-id for block computation, or use --merge.")

    outfile = compute_block(args.task_id, args.Lmax, args.outdir)
    print(f"Finished block -> {outfile}")


if __name__ == "__main__":
    main()
