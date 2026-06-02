
# # Imports


from functools import lru_cache
from pathlib import Path
import pickle
import numpy as np
from sympy.physics.wigner import wigner_3j
from itertools import product


# # Define Functions


#Functions to define C_ells and \Gamma_ells 

def C_ell_primes(ell1, ell2, ell3):
    z = complex(0, 1)
    result = (z)**(ell1 + ell2 + ell3) \
             * np.sqrt(2*ell1 + 1) \
             * np.sqrt(2*ell2 + 1) \
             * np.sqrt(2*ell3 + 1)
    return result

def Gamma_L(L1, L2, L3):
    L = L1 + L2 + L3
    numerator   = (-1)**L
    denominator = np.sqrt(2*L1 + 1) * np.sqrt(2*L2 + 1) * np.sqrt(2*L3 + 1)
    return numerator / denominator


def Cprime_times_Gamma(L1, L2, L3):
    """
    Product C'_ell * Gamma_L with ell_i = L_i.
    """
    return C_ell_primes(L1, L2, L3) * Gamma_L(L1, L2, L3)


def Precompute_Cprime_Gamma(pkl_path="Cprime_Gamma_all.pkl",Lmax=16):
    """
    Precompute C'_ell * Gamma_L for all combinations:
      L1 in [0, Lmax]
      L2 in [0, Lmax]
      L3 in [0, Lmax]
    """
    pkl_path = Path(pkl_path)

    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            data = pickle.load(f)
    else:
        data = {}

    for L1 in range(0, Lmax + 1):
        for L2 in range(0, Lmax + 1):
            for L3 in range(0, Lmax + 1):
                key = (int(L1), int(L2), int(L3))
                if key in data:
                    continue
                val = Cprime_times_Gamma(L1, L2, L3)
                data[key] = complex(val)  # keep as complex explicitly

        # save incrementally so you don't lose progress
        with pkl_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data



#Define the modified Gaunt

@lru_cache(maxsize=None)
def W3j(l1, l2, l3, m1, m2, m3):
    # sympy returns Rational; convert once to float
    return float(wigner_3j(int(l1), int(l2), int(l3),
                           int(m1), int(m2), int(m3)))

@lru_cache(maxsize=None)
def Modified_Gaunt(l1, l2, l3):
    """
    Modified_Gaunt(l1,l2,l3) =
      sqrt((2l1+1)(2l2+1)(2l3+1)/(4π)) * (l1 l2 l3; 0 0 0)
    """
    pref = np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*np.pi))
    return pref  * W3j(l1,l2,l3,0,0,0)

def precompute_Modified_Gaunt (pkl_path="Modified_G_all.pkl",Lmax=16):
    """
    Precompute \mathcal{G} for all combinations:
      L1 in [0, Lmax]
      L2 in [0, Lmax]
      L3 in [0, Lmax]
    """
    pkl_path = Path(pkl_path)

    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            data = pickle.load(f)
    else:
        data = {}

    for L1 in range(0, Lmax + 1):
        for L2 in range(0, Lmax + 1):
            for L3 in range(0, Lmax + 1):
                key = (int(L1), int(L2), int(L3))
                if key in data:
                    continue
                val = Modified_Gaunt(L1, L2, L3)
                data[key] = complex(val)  # keep as complex explicitly

        # save incrementally so you don't lose progress
        with pkl_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data



@lru_cache(maxsize=None)
def Gaunt(l1, l2, l3, m1, m2, m3):
    """
    Gaunt(l1,l2,l3; m1,m2,m3) =
      sqrt((2l1+1)(2l2+1)(2l3+1)/(4π)) *
      (l1 l2 l3; m1 m2 m3)(l1 l2 l3; 0 0 0)
    """
    pref = np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*np.pi))
    return pref * W3j(l1,l2,l3,m1,m2,m3) * W3j(l1,l2,l3,0,0,0)

@lru_cache(maxsize=None)
def modified_H(L, Lp, Lpp, ell):
    """
    Fast evaluation of H-coefficient using selection rules:

    - W3j(L,Lp,Lpp; M,Mp,Mpp) => Mpp = -M - Mp
    - Gaunt(Lpp,ell,ellp; Mpp,m,-mp) requires Mpp + m - mp = 0 => mp = Mpp + m

    So we can remove loops over (Mpp, mp).
    """
    res = 0.0
    phase = (-1) ** (L + Lp + Lpp)

    for M in range(-L, L + 1):
        for Mp in range(-Lp, Lp + 1):

            Mpp = -M - Mp
            if abs(Mpp) > Lpp:
                continue

            for m in range(-ell, ell + 1):

                # ellp triangle: |ell - Lpp| <= ellp <= ell + Lpp
                for ellp in range(abs(ell - Lpp), ell + Lpp + 1):

                    mp = Mpp + m
                    if abs(mp) > ellp:
                        continue

                    res += (phase * W3j(L, Lp, Lpp, M, Mp, Mpp)
                        * Gaunt(Lpp, ell, ellp, Mpp, m, -mp) * Gaunt(L,  Lp,  ellp, M,  Mp,  mp))

    return res


def precompute_Modified_H(pkl_path="Modified_H_all.pkl", Lmax=16, jmax=2):
    """
    Precompute H for all combinations:
      L1,L2,L3 in [0,Lmax],  j in [0,jmax].

    Saves once per L1 (much less I/O, still safe).
    """
    pkl_path = Path(pkl_path)
    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            data = pickle.load(f)
    else:
        data = {}

    for L1 in range(0, Lmax + 1):
        for L2 in range(0, Lmax + 1):
            for L3 in range(0, Lmax + 1):
                for j in range(0, jmax + 1):
                    key = (int(L1), int(L2), int(L3), int(j))
                    if key in data:
                        continue
                    data[key] = complex(modified_H(L1, L2, L3, j))

        # incremental save (per L1)
        with pkl_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data                 


@lru_cache(maxsize=None)
def modified_J(L, Lp, Lpp, ell, ellp):
    """
    Fast evaluation of J-coefficient using m-sum selection rules:

    - W3j(L,Lp,Lpp; M,Mp,Mpp) => Mpp = -M - Mp
    - Gaunt(Lpp,ell,ellpp; Mpp,m,-mpp) => mpp = Mpp + m
    - Gaunt(ellp,ellpp,ellppp; mp,mpp,-mppp) => mp = -mpp + mppp
    - Gaunt(L,Lp,ellppp; M,Mp,mppp) => mppp = -M - Mp

    Combining these gives:
      Mpp = -M-Mp
      mppp = -M-Mp (= Mpp)
      mpp  = Mpp + m
      mp   = -m
    so we can remove loops over (Mpp, mp, mpp, mppp).
    """
    res = 0.0
    phase = (-1) ** (L + Lp + Lpp)

    for M in range(-L, L + 1):
        for Mp in range(-Lp, Lp + 1):

            Mpp = -M - Mp
            if abs(Mpp) > Lpp:
                continue

            # fixed by the last Gaunt
            mppp = -M - Mp  # = Mpp

            for m in range(-ell, ell + 1):

                # fixed by the middle Gaunt m-sum (mp + mpp - mppp = 0)
                mp = -m
                if abs(mp) > ellp:
                    continue

                # ellpp triangle: |ell - Lpp| <= ellpp <= ell + Lpp
                for ellpp in range(abs(ell - Lpp), ell + Lpp + 1):

                    mpp = Mpp + m  # fixed by Gaunt(Lpp,ell,ellpp; Mpp,m,-mpp)
                    if abs(mpp) > ellpp:
                        continue

                    # ellppp triangle: |ellp - ellpp| <= ellppp <= ellp + ellpp
                    for ellppp in range(abs(ellp - ellpp), ellp + ellpp + 1):

                        if abs(mppp) > ellppp:
                            continue

                        res += (phase * W3j(L, Lp, Lpp, M, Mp, Mpp) * Gaunt(Lpp, ell,  ellpp,  Mpp, m,   -mpp)
                            * Gaunt(ellp, ellpp, ellppp, mp,  mpp, -mppp) * Gaunt(L,   Lp,   ellppp,  M,   Mp,  mppp))

    return res


# # Computation of terms 

CG_constant = Precompute_Cprime_Gamma("Cprime_Gamma_all.pkl")

Modified_G = precompute_Modified_Gaunt("Modified_G_all.pkl")

Modified_H = precompute_Modified_H ("Modified_H_all.pkl")