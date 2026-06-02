# # Imports and Definitions

from scipy.special import spherical_jn
import numpy as np
import camb
from camb import model, initialpower
import matplotlib.pyplot as plt


#define the variables
rmin = 0.
rmax = 200.
numr = 200 #this must be type integer.
numri = 80
r = np.linspace(rmin, rmax, numr)+5 #these are intermediate variables that get integrated out in the 4PCF paper
ri = np.linspace(rmin, rmax, numri)+5 #these are the variables the 4PCF depends on (r1, r2, r3)


kmin = 1e-4
kmax = 3.
numk = 200 #change to 1250 for f' integral (to ensure convergence)
k = np.logspace(np.log10(kmin), np.log10(kmax), numk)



#Now get matter power spectra and sigma8 at redshift 0 and 0.8
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)

#Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(kmax=kmax)

#Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)

#Obtain Power Spectrum with pk below
kh, z, Plin_k = results.get_matter_power_spectrum(minkh=1e-4, maxkh=3, npoints = numk)



num_ell = 11 #i.e. ell = 0, 1, 2.

j = np.zeros((num_ell, numr, numk))
jri = np.zeros((num_ell, numri, numk))


for ell in range(num_ell):
    #print('ell =', ell)
    i = 0
    for rval in r:
        j[ell, i, :] = spherical_jn(ell, k*rval)
        i += 1
        #print('i=',i)

for ell in range(num_ell):
    #print('ell =', ell)
    i = 0
    for rval in ri:
        jri[ell, i, :] = spherical_jn(ell, k*rval)
        i += 1        


num_pow = 9 #i.e. 0, 1, 2, 3, 4, 5, 6
k_pow = np.zeros((num_pow, numk))
r_pow = np.zeros((num_pow, numr))
ri_pow = np.zeros((num_pow, numri))


k_pow[0, :] = np.ones(numk) # = k^0 = {1,...,1}
k_pow[1, :] = k # = k^1

r_pow[0, :] = np.ones(numr) # = r^0 = {1,...,1}
r_pow[1, :] = r

ri_pow[0, :] = np.ones(numri)
ri_pow[1, :] = ri

for i in range(1, num_pow - 1):
    #print('i in power calc', i)
    k_pow[i + 1, :] = k_pow[i, :]*k
    r_pow[i + 1, :] = r_pow[i, :]*r
    ri_pow[i + 1, :] = ri_pow[i, :]*ri



dlnk = (np.log(k[1])-np.log(k[0])) #infinitesimal radial element for our wave-vector integrals
dr = (np.max(r)-np.min(r))/len(r) #infinitesimal radial element for our vector integrals
tpi = 1/(2*np.pi**2) # = 1/2*pi^2


# ### Note for the infinitesimal element of k: Since we are in log space we need to find the infinitesimal in this same space. Also, $dk = k\times d\ln(k)$, which means in our integrand below we need to introduce an extra factor of k.
# 


# # Numerical Integral of f 


ell_max = 3
n_max = 3




import numpy as np
import os

# load existing if present
if os.path.exists("f_integrals.npz"):
    f = np.load("f_integrals.npz", allow_pickle=True)["f"].item()
else:
    f = {}

ell_max = 10 # <-- extend here if needed
n_max_new = 5   # <-- extend here if needed

Pk = Plin_k[0]

for n in range(-4, n_max_new + 1):
    if n not in f:
        f[n] = {}
    for ell1 in range(ell_max + 1):
        if ell1 not in f[n]:
            f[n][ell1] = {}
        for ell2 in range(0, ell_max + 1):
            if ell2 in f[n][ell1]:
                continue  # already computed

            f[n][ell1][ell2] = (
                dlnk * tpi *
                np.einsum(
                    'lm,nm,m',
                    j[ell1,:,:],
                    jri[ell2,:,:],
                    k_pow[n+3,:] * Pk
                )
            )

np.savez_compressed("f_integrals.npz", f=f)
print("Updated f_integrals.npz")


