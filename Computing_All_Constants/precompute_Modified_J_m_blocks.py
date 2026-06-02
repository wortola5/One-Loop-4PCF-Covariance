from functools import lru_cache
from pathlib import Path
import argparse
import os
import pickle
import numpy as np
from sympy.physics.wigner import wigner_3j


# ============================================================
# Basic angular building blocks
# ============================================================

@lru_cache(maxsize=None)
def W3j(l1, l2, l3, m1, m2, m3):
    return float(
        wigner_3j(
            int(l1), int(l2), int(l3),
            int(m1), int(m2), int(m3)
        )
    )


@lru_cache(maxsize=None)
def Gaunt(l1, l2, l3, m1, m2, m3):
    """
    Gaunt(l1,l2,l3; m1,m2,m3) =
      sqrt((2l1+1)(2l2+1)(2l3+1)/(4π)) *
      (l1 l2 l3; m1 m2 m3)(l1 l2 l3; 0 0 0)
    """
    pref = np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * np.pi))
    return pref * W3j(l1, l2, l3, m1, m2, m3) * W3j(l1, l2, l3, 0, 0, 0)


# ============================================================
# Modified J with external m, mp included
# ============================================================

@lru_cache(maxsize=None)
def modified_J_m_included(L, Lp, Lpp, ell, ellp, m, mp):
    """
    Fast evaluation of J-coefficient with external magnetic moments m and mp included.

    Selection rules:

    - W3j(L,Lp,Lpp; M,Mp,Mpp) => Mpp = -M - Mp
    - Gaunt(Lpp,ell,ellpp; Mpp,m,-mpp) => mpp = Mpp + m
    - Gaunt(ellp,ellpp,ellppp; mp,mpp,-mppp)
    - Gaunt(L,Lp,ellppp; M,Mp,mppp) => mppp = -M - Mp

    Therefore:
      mppp = Mpp
      mpp  = Mpp + m
      and the middle Gaunt requires
         mp + mpp - mppp = 0
      => mp + (Mpp + m) - Mpp = 0
      => mp = -m

    So this object is identically zero unless mp = -m.
    """
    if abs(m) > ell or abs(mp) > ellp:
        return 0.0 + 0.0j

    if mp != -m:
        return 0.0 + 0.0j

    res = 0.0
    phase = (-1) ** (L + Lp + Lpp)

    for M in range(-L, L + 1):
        for Mp in range(-Lp, Lp + 1):
            Mpp = -M - Mp
            if abs(Mpp) > Lpp:
                continue

            mppp = Mpp

            for ellpp in range(abs(ell - Lpp), ell + Lpp + 1):
                mpp = Mpp + m
                if abs(mpp) > ellpp:
                    continue

                for ellppp in range(abs(ellp - ellpp), ellp + ellpp + 1):
                    if abs(mppp) > ellppp:
                        continue

                    term = (
                        phase
                        * W3j(L, Lp, Lpp, M, Mp, Mpp)
                        * Gaunt(Lpp, ell, ellpp, Mpp, m, -mpp)
                        * Gaunt(ellp, ellpp, ellppp, mp, mpp, -mppp)
                        * Gaunt(L, Lp, ellppp, M, Mp, mppp)
                    )
                    res += term

    return complex(res)


# ============================================================
# Block computation and merging
# ============================================================

def task_id_to_triplet(task_id, Lmax):
    nL = Lmax + 1
    L1 = task_id // (nL * nL)
    rem = task_id % (nL * nL)
    L2 = rem // nL
    L3 = rem % nL
    if not (0 <= L1 <= Lmax and 0 <= L2 <= Lmax and 0 <= L3 <= Lmax):
        raise ValueError(f"task_id={task_id} is out of range for Lmax={Lmax}")
    return int(L1), int(L2), int(L3)


def block_filename(L1, L2, L3):
    return f"Modified_J_m_included_L1_{L1:02d}_L2_{L2:02d}_L3_{L3:02d}.pkl"


def atomic_pickle_dump(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def compute_block(L1, L2, L3, jmax=2, outdir="Modified_J_m_blocks", overwrite=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / block_filename(L1, L2, L3)
    if outpath.exists() and not overwrite:
        print(f"Block already exists, skipping: {outpath}")
        return outpath

    data = {}
    for j in range(0, jmax + 1):
        for jp in range(0, jmax + 1):
            for m in range(-j, j + 1):
                for mp in range(-jp, jp + 1):
                    key = (int(L1), int(L2), int(L3), int(j), int(jp), int(m), int(mp))
                    data[key] = complex(modified_J_m_included(L1, L2, L3, j, jp, m, mp))

    atomic_pickle_dump(data, outpath)
    print(f"Wrote {outpath} with {len(data)} entries")
    return outpath


def merge_blocks(Lmax, outdir="Modified_J_m_blocks", merged_name="Modified_J_m_included.pkl", require_all=True):
    outdir = Path(outdir)
    merged = {}
    missing = []

    for L1 in range(0, Lmax + 1):
        for L2 in range(0, Lmax + 1):
            for L3 in range(0, Lmax + 1):
                path = outdir / block_filename(L1, L2, L3)
                if not path.exists():
                    missing.append(str(path))
                    continue
                with path.open("rb") as f:
                    block = pickle.load(f)
                merged.update(block)

    if missing and require_all:
        raise FileNotFoundError(
            "Missing block files; refusing to merge because require_all=True. "
            f"First missing file: {missing[0]}"
        )

    merged_path = outdir / merged_name
    atomic_pickle_dump(merged, merged_path)
    print(f"Merged {len(merged)} entries into {merged_path}")
    if missing:
        print(f"Warning: {len(missing)} blocks were missing during merge")
    return merged_path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Precompute Modified J with external magnetic number m included, blockwise in (L1,L2,L3)."
    )
    parser.add_argument("--Lmax", type=int, default=16)
    parser.add_argument("--jmax", type=int, default=2)
    parser.add_argument("--task-id", type=int, default=None,
                        help="Array task id mapped to one (L1,L2,L3) block.")
    parser.add_argument("--L1", type=int, default=None)
    parser.add_argument("--L2", type=int, default=None)
    parser.add_argument("--L3", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="Modified_J_m_blocks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Allow merge even if not all blocks are present.")

    args = parser.parse_args()

    if args.merge:
        merge_blocks(
            Lmax=args.Lmax,
            outdir=args.outdir,
            merged_name="Modified_J_m_included.pkl",
            require_all=(not args.allow_missing),
        )
        return

    if args.task_id is not None:
        L1, L2, L3 = task_id_to_triplet(args.task_id, args.Lmax)
    else:
        if args.L1 is None or args.L2 is None or args.L3 is None:
            raise ValueError("Provide either --task-id or all of --L1 --L2 --L3")
        L1, L2, L3 = int(args.L1), int(args.L2), int(args.L3)

    compute_block(
        L1=L1,
        L2=L2,
        L3=L3,
        jmax=args.jmax,
        outdir=args.outdir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
