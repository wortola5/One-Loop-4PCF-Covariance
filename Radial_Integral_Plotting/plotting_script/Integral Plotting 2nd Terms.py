
# # Imports

import matplotlib.pyplot as plt
import scipy as sp
from scipy.special import spherical_jn
import matplotlib.colors as colors
import numpy as np
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import Normalize
from matplotlib import rc
rc('font',size='14',family='serif')
plt.rcParams['pdf.fonttype'] = 3
from pylab import rcParams, cm
rcParams['figure.figsize'] = 5, 3
import camb
from camb import model, initialpower
#from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show


# # Set Up for Plotting


#define the variables
rmin = 0.
rmax = 200. 
numr = 200
numri = 80
r = np.linspace(rmin, rmax, numr)+5
ri = np.linspace(rmin, rmax, numri)+5

num_pow = 5 #i.e. 0, 1, 2, 3, 4
r_pow = np.zeros((num_pow, numr))
ri_pow = np.zeros((num_pow, numri))

r_pow[0, :] = np.ones(numr)
r_pow[1, :] = r

ri_pow[0, :] = np.ones(numri)
ri_pow[1, :] = ri
for i in range(1, num_pow - 1):
    r_pow[i + 1, :] = r_pow[i, :]*r
    ri_pow[i + 1, :] = ri_pow[i, :]*ri



num_pow = 9 
kmin = 1e-4
kmax = 3.
numk = 200
k = np.logspace(np.log10(kmin), np.log10(kmax), numk)

k_pow = np.zeros((num_pow, numk))
k_pow[0, :] = np.ones(numk) # = k^0 = {1,...,1}
k_pow[1, :] = k # = k^1

for i in range(1, num_pow - 1):
    #print('i in power calc', i)
    k_pow[i + 1, :] = k_pow[i, :]*k

num_ell = 7 #i.e. ell = 0, 1, 2.

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

Pk = Plin_k[0,:]



#define dr for integral over r
dr = (np.max(r)-np.min(r))/len(r)
dlnk = (np.log(k[1])-np.log(k[0])) #infinitesimal radial element for our wave-vector integrals
tpi = 1/(2*np.pi**2)

#Define some weights 
U = 100 #Mpc/h
u = 10  #Mpc/h



#define colorbar format
def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)



#define the weigth of the integrals for shape of array (200,80)
r2sqd, r3sqd = np.meshgrid(ri_pow[2,:], r_pow[2,:])
f_wt = r2sqd*r3sqd
f_wt.shape

#define the weigth of the integrals for shape of array (80,80)
r4sqd, r5sqd = np.meshgrid(ri_pow[2,:], ri_pow[2,:])
wti = r4sqd*r5sqd
wti.shape



g0_000 = np.load('/path/to/g0_000.npy')
g0_001 = np.load('/path/to/g0_001.npy')



g0_000.shape


# # Plotting Fully-Coupled $S$ Integrals 


#Compute the S integral in Covariance work for all ang. moments = 0 
S_2nd_00 = dr*np.einsum('il,jl,l,l,l,l',g0_000[:,100,:],g0_000[:,100,:],g0_000[40,100,:],g0_000[40,100,:],g0_000[40,100,:],r_pow[2,:]) 
#Result as a Function of(r_1,r_2)



np.save('S_2nd_00.npy',S_2nd_00)
S_2nd_00.shape



#Compute the S integral in Covariance work for all ang. moments = 0, except the L2 = 1 (corresponds to r2)
S_2nd_01 = dr*np.einsum('il,jl,l,l,l,l',g0_000[:,100,:],g0_001[:,100,:],g0_000[40,100,:],g0_000[40,100,:],g0_000[40,100,:],r_pow[2,:]) 
#Result as a Function of(r_1,r_2)

np.save('S_2nd_01.npy',S_2nd_01)
S_2nd_01.shape



# g0_000_W = g0_000[:,100,:]*f_wt.transpose() #function of (r_i,r')
# g0_001_W = g0_001[:,100,:]*f_wt.transpose()


S_00_W = S_2nd_00*wti
S_01_W = S_2nd_01*wti



fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,  sharey=True, sharex=True,figsize=(6,8))

extent = rmin,rmax,rmin,rmax #Gives range we want it to appear on x,y axis 
vmin = -30
vmax =30
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
#norm = colors.TwoSlopeNorm(vmin=minmin, vcenter=0, vmax=maxmax)
im3 = ax1.imshow(g0_000_W.transpose()/U**2, origin='lower', cmap=cm.RdBu_r,norm=norm,extent=extent) 
im4 = ax2.imshow(g0_001_W.transpose()/U**2, origin='lower', cmap=cm.RdBu_r,norm=norm,extent=extent) 
ax1.set_title(r'$r^{\prime 2} r^{\prime \prime 2}/u^{4} \;g^{[0]}_{0,0,0} (U,r^{\prime},r^{\prime \prime})$') 
ax2.set_title(r'$r^{\prime 2} r^{\prime \prime 2}/u^{4} \;g^{[0]}_{0,0,1} (U,r^{\prime},r^{\prime \prime})$')
ax1.set_ylabel(r'$r^{\prime}\;{\rm [Mpc}/h {\rm ]}$')
ax2.set_ylabel(r'$r^{\prime}\;{\rm [Mpc}/h {\rm ]}$')
ax2.set_xlabel(r'$r^{\prime \prime}\;{\rm [Mpc}/h {\rm ]}$')
#xticks(np.arange(0, 250, step=50)) 
#yticks(np.arange(0, 250, step=50))
fig.subplots_adjust(right=0.65)
cbar_ax = fig.add_axes([0.7, 0.05, 0.04, 0.85])
fig.colorbar(im4, cax=cbar_ax)#,format=ticker.FuncFormatter(fmt))
plt.savefig('gint0.pdf')
plt.show()



fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,  sharey=True, sharex=True,figsize=(6,8))

extent = rmin,rmax,rmin,rmax #Gives range we want it to appear on x,y axis 
vmin = -0.0002
vmax =0.0002
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
#norm = colors.TwoSlopeNorm(vmin=minmin, vcenter=0, vmax=maxmax)
im3 = ax1.imshow(S_00_W/U**2, origin='lower', cmap=cm.RdBu_r,norm=norm,extent=extent) 
im4 = ax2.imshow(S_01_W.transpose()/U**2, origin='lower', cmap=cm.RdBu_r,norm=norm,extent=extent) 
ax1.set_title(r'$r_1^2 r_2^2/u^{4} \;S^{\{0\}}_{\{0\}} (U,U,r_1,U,r_2,U,U,U)$',fontsize = 12) 
ax2.set_title(r'$r_1^2 r_2^2/u^{4} \;S^{\{0\}}_{\{0\},L_2 = 1} (U,U,r_1,U,r_2,U,U,U)$',fontsize = 12)
ax1.set_ylabel(r'$r_1\;{\rm [Mpc}/h {\rm ]}$')
ax2.set_ylabel(r'$r_1\;{\rm [Mpc}/h {\rm ]}$')
ax2.set_xlabel(r'$r_2\;{\rm [Mpc}/h {\rm ]}$')
#xticks(np.arange(0, 250, step=50)) 
#yticks(np.arange(0, 250, step=50))
fig.subplots_adjust(right=0.65)
cbar_ax = fig.add_axes([0.7, 0.05, 0.04, 0.85])
fig.colorbar(im4, cax=cbar_ax,format=ticker.FuncFormatter(fmt))
plt.savefig('Sint.pdf')
plt.show()


# # Plotting Partially-Coupled $S$ Integrals 

# ## Radial integral of PC-H-H



#Compute the S integral for PC-H-H with all L=0
S_00_PC_HH = dr*np.einsum('il,jl,l,l',g0_000[:,100,:],g0_000[:,100,:],g0_000[40,100,:],r_pow[2,:])
#Result as a Function of(r_2,r_3)

np.save('S_00_PC_HH.npy',S_00_PC_HH)
S_00_PC_HH.shape




#Compute the S integral for PC-H-H with all L_3=1
S_01_PC_HH = dr*np.einsum('il,jl,l,l',g0_000[:,100,:],g0_001[:,100,:],g0_000[40,100,:],r_pow[2,:]) 
#Result as a Function of(r_2,r_3)

np.save('S_01_PC_HH.npy',S_01_PC_HH)
S_01_PC_HH.shape




#Weight the radial integrals
S_00_PC_HH_W = S_00_PC_HH*wti
S_01_PC_HH_W = S_01_PC_HH*wti




fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,  sharey=True, sharex=True,figsize=(6,8))

extent = rmin,rmax,rmin,rmax #Gives range we want it to appear on x,y axis 
vmin = -40
vmax =40
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
#norm = colors.TwoSlopeNorm(vmin=minmin, vcenter=0, vmax=maxmax)
im3 = ax1.imshow(S_00_PC_HH_W/U**2, origin='lower', cmap=cm.RdBu_r,norm=norm,extent=extent) 
im4 = ax2.imshow(S_01_PC_HH_W.transpose()/U**2, origin='lower', cmap=cm.RdBu_r,norm=norm,extent=extent) 
ax1.set_title(r'$r_2^2 r_3^2/u^{4} \;S^{\{0\}}_{\{0\},(\rm PC-H-H)} (U,U,r_2,U,r_3,U)$',fontsize = 12) 
ax2.set_title(r'$r_2^2 r_3^2/u^{4} \;S^{\{0\}}_{\{0\},L_3 = 1,(\rm PC-H-H)} (U,U,r_2,U,r_3,U)$',fontsize = 12)
ax1.set_ylabel(r'$r_2\;{\rm [Mpc}/h {\rm ]}$')
ax2.set_ylabel(r'$r_2\;{\rm [Mpc}/h {\rm ]}$')
ax2.set_xlabel(r'$r_3\;{\rm [Mpc}/h {\rm ]}$')
#xticks(np.arange(0, 250, step=50)) 
#yticks(np.arange(0, 250, step=50))
fig.subplots_adjust(right=0.65)
cbar_ax = fig.add_axes([0.7, 0.05, 0.04, 0.85])
fig.colorbar(im4, cax=cbar_ax,format=ticker.FuncFormatter(fmt))
plt.savefig('S_PC_HH_int.pdf')
plt.show()


# ## Radial Integral of PC-P-S


g0_000_80 = dlnk*tpi*np.einsum('lm,nm,jm,m', jri[0,:, :], jri[0,:, :], jri[0,:,:],  k_pow[3, :]*Pk)
g0_001_80 = dlnk*tpi*np.einsum('lm,nm,jm,m', jri[0,:, :], jri[0,:, :], jri[1,:10,:],  k_pow[3, :]*Pk)

print(g0_001_80.shape, flush = True)




#Compute the S integral for PC-P-S with all L=0
#S_00_PC_PS = dr*np.einsum('ikl,l',g0_000[:,:,:],r_pow[2,:]) 
S_00_PC_PS = dr*np.einsum('ikl,l',g0_000_80[:,:,:],ri_pow[2,:]) 
#Result as a Function of(r'_3,r_3)

np.save('S_00_PC_PS.npy',S_00_PC_PS)
S_00_PC_PS.shape




#Compute the S integral for PC-P-S with all L_3=1
# S_01_PC_PS = dr*np.einsum('ikl,l',g0_001[:,:,:],r_pow[2,:]) 
S_01_PC_PS = dr*np.einsum('ikl,l',g0_001_80[:,:,:],ri_pow[2,:]) 
#Result as a Function of(r'_3,r_3)

np.save('S_01_PC_PS.npy',S_01_PC_PS)
S_01_PC_PS.shape




#Weight the radial integrals
# S_00_PC_PS_W = S_00_PC_PS*f_wt.transpose()
# S_01_PC_PS_W = S_01_PC_PS*f_wt.transpose()

S_00_PC_PS_W = S_00_PC_PS*wti
S_01_PC_PS_W = S_01_PC_PS*wti




fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,  sharey=True, sharex=True,figsize=(6,8))

extent = rmin,rmax,rmin,rmax #Gives range we want it to appear on x,y axis 
vmin = -70000000
vmax =70000000
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
#norm = colors.TwoSlopeNorm(vmin=minmin, vcenter=0, vmax=maxmax)
im3 = ax1.imshow(S_00_PC_PS_W/U**2, origin='lower', cmap=cm.RdBu_r,norm=norm,extent=extent) 
im4 = ax2.imshow(S_01_PC_PS_W/U**2, origin='lower', cmap=cm.RdBu_r,norm=norm,extent=extent) 
ax1.set_title(r'$r_3^2 r_3^{\prime 2}/u^{4} \;S^{\{0\}}_{\{0\},(\rm PC-P-S)} (r_3,r_3^{\prime})$',fontsize = 12) 
ax2.set_title(r'$r_3^2 r_3^{\prime 2}/u^{4} \;S^{\{0\}}_{L_3 = 0,L_3^{\prime}=1,(\rm PC-P-S)} (r_3,r_3^{\prime})$',fontsize = 12)
ax1.set_ylabel(r'$r_3\;{\rm [Mpc}/h {\rm ]}$')
ax2.set_ylabel(r'$r_3\;{\rm [Mpc}/h {\rm ]}$')
ax2.set_xlabel(r'$r_3^{\prime}\;{\rm [Mpc}/h {\rm ]}$')
#xticks(np.arange(0, 250, step=50)) 
#yticks(np.arange(0, 250, step=50))
fig.subplots_adjust(right=0.65)
cbar_ax = fig.add_axes([0.7, 0.05, 0.04, 0.85])
fig.colorbar(im4, cax=cbar_ax,format=ticker.FuncFormatter(fmt))
#plt.savefig('S_PC_PS_int.pdf')
plt.show()




fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,  sharey=True, sharex=True,figsize=(6,8))

extent = rmin,rmax,rmin,rmax #Gives range we want it to appear on x,y axis 
vmin = -70000000
vmax =70000000
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
#norm = colors.TwoSlopeNorm(vmin=minmin, vcenter=0, vmax=maxmax)
im3 = ax1.imshow(S_00_PC_PS_W/U**2, origin='lower', cmap=cm.RdBu_r,norm=norm,extent=extent) 
im4 = ax2.imshow(S_01_PC_PS_W/U**2, origin='lower', cmap=cm.RdBu_r,norm=norm,extent=extent) 
ax1.set_title(r'$r_3^2 r_3^{\prime 2}/u^{4} \;S^{\{0\}}_{\{0\},(\rm PC-P-S)} (r_3,r_3^{\prime})$',fontsize = 12) 
ax2.set_title(r'$r_3^2 r_3^{\prime 2}/u^{4} \;S^{\{0\}}_{L_3 = 0,L_3^{\prime}=1,(\rm PC-P-S)} (r_3,r_3^{\prime})$',fontsize = 12)
ax1.set_ylabel(r'$r_3\;{\rm [Mpc}/h {\rm ]}$')
ax2.set_ylabel(r'$r_3\;{\rm [Mpc}/h {\rm ]}$')
ax2.set_xlabel(r'$r_3^{\prime}\;{\rm [Mpc}/h {\rm ]}$')
#xticks(np.arange(0, 250, step=50)) 
#yticks(np.arange(0, 250, step=50))
fig.subplots_adjust(right=0.65)
cbar_ax = fig.add_axes([0.7, 0.05, 0.04, 0.85])
fig.colorbar(im4, cax=cbar_ax,format=ticker.FuncFormatter(fmt))
plt.savefig('S_PC_PS_int.pdf')
plt.show()






