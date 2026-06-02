#!/usr/bin/env python3

from functools import lru_cache
from pathlib import Path
import argparse
import pickle
import numpy as np
from sympy.physics.wigner import wigner_3j


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
    pref = np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * np.pi))
    return pref * W3j(l1, l2, l3, m1, m2, m3) * W3j(l1, l2, l3, 0, 0, 0)


@lru_cache(maxsize=None)
def L_tensor(j1, j2, j3, ellp2, L1, L2, L3, m1, m2, m3, mp2):
    """
    Computes the L-tensor coefficient for fixed external magnetic numbers.

    Starting from the screenshot expression, the selection rules imply:
      M3   = -M1 - M2
      m    =  M1 + M2 = -M3
      m'   =  0
      m''  =  m1 + m2
      m(3) =  m3 + mp2

    The whole object is zero unless:
      m1 + m2 + m3 + mp2 = 0
    """
    if (m1 + m2 + m3 + mp2) != 0:
        return 0.0 + 0.0j

    phase = (-1) ** (int(L1) + int(L2) + int(L3))
    res = 0.0 + 0.0j

    mpp = int(m1 + m2)
    m3bar = int(m3 + mp2)

    ellpp_min = abs(int(j1) - int(j2))
    ellpp_max = int(j1) + int(j2)

    ell3_min = abs(int(j3) - int(ellp2))
    ell3_max = int(j3) + int(ellp2)

    ell_min = abs(int(L1) - int(L2))
    ell_max = int(L1) + int(L2)

    for M1 in range(-int(L1), int(L1) + 1):
        for M2 in range(-int(L2), int(L2) + 1):
            M3 = -M1 - M2
            if abs(M3) > int(L3):
                continue

            m = M1 + M2

            for ell in range(ell_min, ell_max + 1):
                if abs(m) > ell:
                    continue

                G_a = Gaunt(L1, L2, ell, M1, M2, -m)
                if G_a == 0:
                    continue

                w = W3j(L1, L2, L3, M1, M2, M3)
                if w == 0:
                    continue

                ellp_min_1 = abs(int(ell) - int(L3))
                ellp_max_1 = int(ell) + int(L3)

                for ellpp in range(ellpp_min, ellpp_max + 1):
                    if abs(mpp) > ellpp:
                        continue

                    G_c = Gaunt(j1, j2, ellpp, m1, m2, -mpp)
                    if G_c == 0:
                        continue

                    for ell3 in range(ell3_min, ell3_max + 1):
                        if abs(m3bar) > ell3:
                            continue

                        G_d = Gaunt(j3, ellp2, ell3, m3, mp2, -m3bar)
                        if G_d == 0:
                            continue

                        ellp_min_2 = abs(int(ell3) - int(ellpp))
                        ellp_max_2 = int(ell3) + int(ellpp)

                        ellp_min = max(ellp_min_1, ellp_min_2)
                        ellp_max = min(ellp_max_1, ellp_max_2)
                        if ellp_min > ellp_max:
                            continue

                        for ellp in range(ellp_min, ellp_max + 1):
                            G_b = Gaunt(ell, L3, ellp, m, M3, 0)
                            if G_b == 0:
                                continue

                            G_e = Gaunt(ell3, ellpp, ellp, m3bar, mpp, 0)
                            if G_e == 0:
                                continue

                            res += phase * w * G_a * G_b * G_c * G_d * G_e

    return complex(res)


def decode_task_id(task_id, lmax):
    side = lmax + 1
    L1 = task_id // (side * side)
    rem = task_id % (side * side)
    L2 = rem // side
    L3 = rem % side
    return L1, L2, L3


def compute_block(L1, L2, L3, jmax=2, Lcutoff=10):
    data = {}

    for j1 in range(0, jmax + 1):
        for j2 in range(0, jmax + 1):
            for j3 in range(0, jmax + 1):
                for ellp2 in range(0, Lcutoff + 1):
                    for m1 in range(-j1, j1 + 1):
                        for m2 in range(-j2, j2 + 1):
                            for m3 in range(-j3, j3 + 1):
                                for mp2 in range(-ellp2, ellp2 + 1):
                                    if (m1 + m2 + m3 + mp2) != 0:
                                        continue

                                    key = (
                                        int(j1), int(j2), int(j3), int(ellp2),
                                        int(L1), int(L2), int(L3),
                                        int(m1), int(m2), int(m3), int(mp2)
                                    )
                                    data[key] = complex(
                                        L_tensor(j1, j2, j3, ellp2, L1, L2, L3, m1, m2, m3, mp2)
                                    )
    return data


def save_block(data, outdir, L1, L2, L3):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    final_path = outdir / f"L_tensor_L1_{L1:02d}_L2_{L2:02d}_L3_{L3:02d}.pkl"
    tmp_path = outdir / f".tmp_L_tensor_L1_{L1:02d}_L2_{L2:02d}_L3_{L3:02d}.pkl"

    with tmp_path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    tmp_path.replace(final_path)
    return final_path


def merge_blocks(outdir, merged_name="L_tensor_all.pkl"):
    outdir = Path(outdir)
    merged = {}
    block_files = sorted(outdir.glob("L_tensor_L1_*_L2_*_L3_*.pkl"))

    for fp in block_files:
        with fp.open("rb") as f:
            block = pickle.load(f)
        merged.update(block)

    merged_path = outdir / merged_name
    tmp_path = outdir / f".tmp_{merged_name}"
    with tmp_path.open("wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(merged_path)
    return merged_path


def main():
    parser = argparse.ArgumentParser(description="Precompute L_tensor in (L1,L2,L3) blocks.")
    parser.add_argument("--Lmax", type=int, default=10)
    parser.add_argument("--jmax", type=int, default=2)
    parser.add_argument("--Lcutoff", type=int, default=10)
    parser.add_argument("--L1", type=int, default=None)
    parser.add_argument("--L2", type=int, default=None)
    parser.add_argument("--L3", type=int, default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="L_tensor_blocks_ellp2")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--merged-name", type=str, default="L_tensor_all.pkl")
    args = parser.parse_args()

    if args.merge:
        merged_path = merge_blocks(args.outdir, args.merged_name)
        print(f"Merged blocks written to: {merged_path}")
        return

    if args.task_id is not None:
        L1, L2, L3 = decode_task_id(args.task_id, args.Lmax)
    else:
        if args.L1 is None or args.L2 is None or args.L3 is None:
            raise ValueError("Provide either --task-id or all of --L1 --L2 --L3")
        L1, L2, L3 = args.L1, args.L2, args.L3

    data = compute_block(L1=L1, L2=L2, L3=L3, jmax=args.jmax, Lcutoff=args.Lcutoff)
    final_path = save_block(data, args.outdir, L1, L2, L3)
    print(f"Finished block (L1,L2,L3)=({L1},{L2},{L3})")
    print(f"Saved to: {final_path}")


if __name__ == "__main__":
    main()