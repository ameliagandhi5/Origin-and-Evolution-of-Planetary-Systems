#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:57:45 2026

@author: ameliagandhi
"""

import numpy as np 
import matplotlib.pyplot as plt 
from astrotools.disk.structure import * 
from astrotools.disk.viscous import * 
import pandas as pd 
from scipy.optimize import curve_fit

# define fit 
def power_fit(x, a, b):
    fit = a*x**b
    return fit 

#%%
# 1. BUILD THE MODULE 
# -------------------

### test scale height at 1 AU 
H_test = scale_height(c.AU, 280, m_star=c.M_SUN, mu=c.MU_DISK)/c.AU
print(f'Scale Height at 1 AU: {H_test:.4f} AU')

### test scale height/r falling as r^1/4
r = np.arange(.1, 30, .1)*c.AU # in m
T = 280*(r/c.AU)**(-.5)

H_test = scale_height(r, T, m_star=c.M_SUN, mu=c.MU_DISK)/c.AU
fig, ax = plt.subplots(1, 1)

ax.loglog(r, H_test/r, c='green', linewidth=3, label='calculated H/r')
ax.loglog(r, 3.58e-16*r**(1/4), c='k', linestyle='-.', label=r'$r^{1/4}$')

# prettify 
ax.set_xlabel('Distance r (AU)')
ax.set_ylabel('Scale Height per Distance H/r')
ax.legend(frameon=False)

### test sed with optically thick disk 
distance = 1*c.PC
wvl = np.logspace(-5, -2, 100) # m 
nu = c.C_LIGHT/wvl
r_edges = radial_grid(0.01, 1e4, n=2000)*c.AU

# find the index where we are at 100 microns 
nu_mask = np.where(nu == c.C_LIGHT/100e-6)[0][0] 

# test sed for slope = -3/4
T_func = lambda r: temperature(r, t_1au=c.T_DISK_1AU, slope=-3/4)
sed_test_34 = sed(nu, r_edges, T_func, distance, sigma_dust=None, kappa=None, cos_i=1.0)

# find the slope with dlog(nu*sed)/dlog(nu)
slope = (np.log(nu[nu_mask + 2]*sed_test_34[nu_mask + 2]) - np.log(nu[nu_mask - 1]*sed_test_34[nu_mask - 1]))/(np.log(nu[nu_mask + 2]) - np.log(nu[nu_mask - 1]))
print(f'Slope with T=r^-3/4: {slope:.3f}')

# test sed for slope = -3/7
T_func = lambda r: temperature(r, t_1au=c.T_DISK_1AU, slope=-3/7)
sed_test_37 = sed(nu, r_edges, T_func, distance, sigma_dust=None, kappa=None, cos_i=1.0)

# find the slope with dlog(nu*sed)/dlog(nu)
slope = (np.log(nu[nu_mask + 2]*sed_test_37[nu_mask + 2]) - np.log(nu[nu_mask - 1]*sed_test_37[nu_mask - 1]))/(np.log(nu[nu_mask + 2]) - np.log(nu[nu_mask - 1]))
print(f'Slope with T=r^-3/7: {slope:.3f}')

### test the optically thin disk 
kappa = lambda nu: dust_opacity(nu, kappa_0=0.23, lambda_0=1.3e-3, beta=1.0) # used chatgpt to help me define a callable function (with lambda) to put in the sed function 
T_func = lambda r: temperature(r, t_1au=20, slope=0) # isothermal temperature at 20 K 
sigma_dust = lambda r: surface_density(r, sigma_1au=c.SIGMA_GAS_1AU, slope=c.SIGMA_SLOPE)

sed_test_thin = sed(nu, r_edges, T_func, distance, sigma_dust=sigma_dust, kappa=kappa, cos_i=1.0)

# make the analytic sed 
r_mid = np.sqrt(r_edges[:-1] * r_edges[1:])
area = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)

kappa_nu = dust_opacity(nu, kappa_0=0.23, lambda_0=1.3e-3, beta=1.0)
B_nu = planck_nu(nu[:,None], 20)
M_dust = np.sum(sigma_dust(r_mid) * area)

sed_test_analytic = kappa_nu * B_nu * 20 * M_dust /distance**2

# plot the two seds to compare 
fig, ax = plt.subplots(1, 1)
ax.loglog(nu, sed_test_thin, c='green', linewidth=3, label='calculated SED for a thin disk')
ax.loglog(nu, sed_test_thin, c='k', linestyle='-.', label=r"$F_\nu = \frac{\kappa_\nu B_\nu(T) M_{\mathrm{dust}}}{d^2}$")
ax.set_ylabel(r'Flux $(W m^{-2} Hz^{-1})$')
ax.set_xlabel('Frequency (Hz)')
ax.set_title('SED of an Optically Thin Disk')
ax.legend(frameon=False)

### test ring evolution 
r0 = 10*c.AU 
grid = make_grid(1*c.AU, 1000*c.AU, 40*3) # used chat gpt to understand that each decade refers to an order of magnitude increase in r 
viscosity = np.full(grid.n, 1e11)
m_ring = c.M_SUN
# convert tau into timesteps, then find the dt with cfl_timestep
tau_start = 0.01
tau_end = 0.1

t_start = tau_start * r0**2 / (12 * 1e11)
t_end = tau_end * r0**2 / (12 * 1e11)

dt = cfl_timestep(grid, 1e11)

# define initial ring with the ring solution 
sigma0 = ring_solution(grid.r, t_start, m_ring, r0, 1e11) # picked a random mass for the ring using chat gpt 

t, sigma, mass = evolve(sigma0, grid, viscosity, t_end - t_start, dt)

sigma_analytic = ring_solution(grid.r, t_end, m_ring, r0, 1e11)

# find the error at every r compared to the analytic solution 
difference = np.abs(sigma[-1] - sigma_analytic)/np.max(sigma_analytic)*100

# find the max error and plot 
max_error = np.max(difference)
print('Maximum error:', max_error)


#%%
# 2. What the spectrum measures
# -----------------------------

# part a 
# ------

T_star = 4000 # K 
R_star = 2*c.R_SUN
M_star = 1*c.M_SUN
d = 150 * c.PC

wvl = np.arange(1e-6, 3e-3, .1e-6) # m 
nu = c.C_LIGHT/wvl

r_edges = radial_grid(0.1, 100, n=2000)*c.AU

# photosphere flux 
photosphere_F_nu = np.pi* R_star**2 * planck_nu(nu, T_star)/(d**2)

# sed for first profile (-3/4)
q_1 = -3/4
T_1 = lambda r: temperature(r, t_1au=81, slope=q_1)
sed_1 = sed(nu, r_edges, T_1, d)


# sed for first profile (-3/7) 
q_2 = -3/7
T_2 = lambda r: temperature(r, t_1au=160, slope=q_2)
sed_2 = sed(nu, r_edges, T_2, d)

# sum the seds with the photosphere 
summed_sed1 = nu*sed_1 + nu*photosphere_F_nu
summed_sed2 = nu*sed_2 + nu*photosphere_F_nu

# plot the two SED profiles 
fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(8, 5))

ax[0].loglog(wvl*1e6, nu*photosphere_F_nu, c='goldenrod', label=r'$F_{\nu}$ of the photosphere')
ax[1].loglog(wvl*1e6, nu*photosphere_F_nu, c='goldenrod', label=r'$F_{\nu}$ of the photosphere')
ax[0].loglog(wvl*1e6, nu*sed_1, c='k', linestyle='--', label='flat disk')
ax[1].loglog(wvl*1e6, nu*sed_2, c='k', label='flared disk')
ax[0].loglog(wvl*1e6, summed_sed1, c='green', linestyle='--', label=r'flat disk + $F_{\nu}$')
ax[1].loglog(wvl*1e6, summed_sed2, c='green', label=r'flared disk + $F_{\nu}$')

# find the slope bewteen 2.2 and 24 microns 
min_index = np.where(np.isclose(wvl, 2.2e-6))[0][0]
max_index = np.where(np.isclose(wvl, 24e-6))[0][0]

diff_log_wvl = np.log((wvl)[max_index]) - np.log((wvl)[min_index])

slope_1 = (np.log(summed_sed1[max_index]) - np.log(summed_sed1[min_index])) / diff_log_wvl
slope_2 = (np.log(summed_sed2[max_index]) - np.log(summed_sed2[min_index])) / diff_log_wvl


# plot this on each plot 
ax[0].loglog([(wvl*1e6)[min_index], (wvl*1e6)[max_index]], [summed_sed1[min_index], summed_sed1[max_index]], c='red', label=f'α = {slope_1:.2f}') 
ax[1].loglog([(wvl*1e6)[min_index], (wvl*1e6)[max_index]], [summed_sed2[min_index], summed_sed2[max_index]], c='red', label=f'α = {slope_2:.2f}') 


# prettify
ax[0].set_xlabel(r'Wavelength $(\mu m)$')
ax[1].set_xlabel(r'Wavelength $(\mu m)$')
ax[0].set_ylabel(r'Flux  $\nu F_{\nu} \, (W m^{-2})$')
ax[0].legend(frameon=False)
ax[1].legend(frameon=False)

# compare the slopes with -(4 - 2/q)
alpha_1_analytic = -(4 + 2/q_1)
alpha_2_analytic = -(4 + 2/q_2)

print('analytical slope 1:', alpha_1_analytic)
print('analytical slope 2:', alpha_2_analytic)


# part b 
# ------ 

# compute F_nu 
freq = c.C_LIGHT/1.3e-3
r_edges = radial_grid(0.1, 30, n=2000)*c.AU

def sigma_dust(r): 
    return 0.01*surface_density(r, sigma_1au=1.7e4, slope=-3/2)

# compute F_nu with oour sed function 
F_nu_survey = sed(np.array([freq]), r_edges, T_2, d, sigma_dust=sigma_dust, kappa=dust_opacity)
print('flux:', F_nu_survey[0])

kappa = dust_opacity(freq, kappa_0=0.23, lambda_0=1.3e-3, beta=1.0)
B = planck_nu(freq, 20)

# compute dust mass 
M_dust_survey= F_nu_survey*d**2 / (kappa*B)
print('Dust mass:', M_dust_survey/c.M_SUN, 'solar masses')

# define a new function to recompute the flux so that I_v = t_v B_v
def sed_thin(nu, r_edges, T_func, distance, sigma_dust=None, kappa=None, cos_i=1.0):
    r_mid = np.sqrt(r_edges[:-1] * r_edges[1:])
    area = np.pi*(r_edges[1:]**2 - r_edges[:-1]**2)
    T = T_func(r_mid)
    B_nu = planck_nu(nu[:,None], T[None,:]) # T_func
                  
    if sigma_dust is None: 
        I_nu = B_nu 
        
    else: 
        tau_nu = kappa(nu)[:,None] * sigma_dust(r_mid)[None,:] / cos_i
        I_nu = tau_nu*B_nu # I changed this! 
        
    F_nu = (cos_i / distance**2) * np.sum(area*I_nu, axis=1)
    return F_nu

# compute F_nu with the new sed function 
F_nu_thin = sed_thin(np.array([freq]), r_edges, T_2, d, sigma_dust=sigma_dust, kappa=dust_opacity)
print('flux:', F_nu_thin[0])

# compute dust mass 
M_dust_thin = F_nu_thin*d**2 / (kappa*B)
print('Dust mass:', M_dust_thin/c.M_SUN, 'solar masses')

factor = M_dust_thin/M_dust_thick
print('Factor:', factor)

#%%
# 3. How a ring spreads
# ---------------------

# part a 
# ------

# make a resolution for 20 - 80 cells per decade 
cells_per_decade_list = np.arange(20, 81, 1)
max_err_list = [] 

# loop through each of the resolutions 
for cpd in cells_per_decade_list: 
    
    # define initial conditions 
    r0 = 10*c.AU 
    grid = make_grid(1*c.AU, 1000*c.AU, int(cpd*3))
    viscosity = np.full(grid.n, 1e11)
    m_ring = c.M_SUN
    
    # convert tau into timesteps, then find the dt with cfl_timestep
    tau_start = 0.01
    tau_end = 0.1
    
    t_start = tau_start * r0**2 / (12 * 1e11)
    t_end = tau_end * r0**2 / (12 * 1e11)
    
    dt = cfl_timestep(grid, 1e11)
    
    # define initial ring with the ring solution 
    sigma0 = ring_solution(grid.r, t_start, m_ring, r0, 1e11) # picked a random mass for the ring using chat gpt 
    
    t, sigma, mass = evolve(sigma0, grid, viscosity, t_end - t_start, dt)
    
    sigma_analytic = ring_solution(grid.r, t_end, m_ring, r0, 1e11)
    
    # find the error at every r compared to the analytic solution 
    difference = np.abs(sigma[-1] - sigma_analytic)/np.max(sigma_analytic)*100
    
    # find the max error and plot 
    max_error = np.max(difference)
    max_err_list.append(max_error)
    
fig, ax = plt.subplots(1, 1)
ax.scatter(cells_per_decade_list, max_err_list, c='k', s=3)

# find the best fit 
popt, pcov = curve_fit(power_fit, cells_per_decade_list, max_err_list, p0=[1,1])

# find order of convergence 
fit = popt[0]*cells_per_decade_list**popt[1]
ax.plot(cells_per_decade_list, fit, c='green', label=r'$x^{-2}$')
print('Order of convergence:', popt[1]) 

# interpolate the cells per decade we need for 1% accuracy 
cells_for_1_percent = (1 / popt[0])**(1 / popt[1])
print('Resolution needed for 1% accuracy:', cells_for_1_percent)

# prettify 
ax.set_ylabel('Maximum Error (%)')
ax.set_xlabel('Cells Per Decade')
ax.legend(frameon=False)

# part b 
# ------

# define initial conditions 
r0 = 10*c.AU 
grid = make_grid(1*c.AU, 1000*c.AU, int(40*3))
viscosity = np.full(grid.n, 1e11)
m_ring = c.M_SUN

# convert tau into timesteps, then find the dt with cfl_timestep
tau_start = 0.01
tau_end = 0.1

t_start = tau_start * r0**2 / (12 * 1e11)
t_end = tau_end * r0**2 / (12 * 1e11)

dt = 1.5 * cfl_timestep(grid, 1e11, safety=1) # divide by 200 to add more steps 

# define initial ring with the ring solution 
sigma0 = ring_solution(grid.r, t_start, m_ring, r0, viscosity)

t, sigma, mass = evolve(sigma0, grid, viscosity, t_end - t_start, dt)
taus = t * (12 * 1e11) / r0**2
sigma_min = np.min(sigma, axis=1)

# plot sigma(r)  
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# plot tau vs sigma max and average 
ax[0].plot(taus, sigma_min, label='Minimum Sigma')

# determine the tau where sigma fails 
tau_index =  np.where(sigma_min < 0)[0][0] 
ax[0].axvline(taus[tau_index], c='red', label=f'τ: {taus[tau_index]:.3f}')
print(f'Run starts failing at tau {taus[tau_index]}')

# plot solutions as we approach this tau and afterwards 
for i in [87, 83, 85]:
    ax[1].plot(grid.r/c.AU, sigma[i], label=f'Tau {taus[i]:.4f}')


# radius where sigma fails 
negative_sigma_indices = np.where(sigma[87] < 0)[0][-1]
ax[1].axvline((grid.r/c.AU)[negative_sigma_indices], c='red', label=f'{(grid.r/c.AU)[negative_sigma_indices]:.2f} AU')
print(f'Radii where sigma fails: R < {(grid.r/c.AU)[negative_sigma_indices]:.2f} AU')

# prettify 
ax[0].set_xlabel(r'$\tau$')
ax[1].set_xlabel('Radius (AU)')
ax[0].set_ylabel(r'$\Sigma$')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[0].legend(frameon=False) 
ax[1].legend(frameon=False) 

# fails because it is negative and starts oscillating 
# HELP!!!! Explain why the inner edge sets the timestep, and how the number of steps needed for a fixed
# duration scales with the number of cells per decade.

#%%
# part c 
# ------
# this takes a little bit to run! 

### for original conditions

# define initial conditions 
r0 = 10*c.AU 
grid = make_grid(1*c.AU, 1000*c.AU, int(40*3))
viscosity = np.full(grid.n, 1e11)
m_ring = c.M_SUN

# convert tau into timesteps, then find the dt with cfl_timestep
tau_start = 0.01
tau_end = np.linspace(.1, .5, 100) 

# loop through each tau end and find the maximum error 
max_errs = [] 

for tau_end_i in tau_end: 
    
    # define initial and final times 
    t_start = tau_start * r0**2 / (12 * 1e11)
    t_end = tau_end_i * r0**2 / (12 * 1e11)
    
    dt = cfl_timestep(grid, 1e11)
    
    # define initial ring with the ring solution, evolve solution, find analytic solution
    sigma0 = ring_solution(grid.r, t_start, m_ring, r0, 1e11) # picked a random mass for the ring using chat gpt 
    t, sigma, mass = evolve(sigma0, grid, viscosity, t_end - t_start, dt)
    sigma_analytic = ring_solution(grid.r, t_end, m_ring, r0, 1e11)
    
    # find the error at every r compared to the analytic solution 
    max_err = np.max(np.abs(sigma[-1] - sigma_analytic)/np.max(sigma_analytic)*100)
    
    max_errs.append(max_err)
    
# plot errors vs tau 
fig, ax = plt.subplots(1, 1)
ax.plot(tau_end, max_errs, c='k', label='inner edge at 1 AU')

# find where the errors exceed 1%, find tau where that index is, rerun for that tau 
first_i_gt1 = np.where(np.array(max_errs) >= 1)[0][0]
tau_end1 = tau_end[first_i_gt1]
print('tau where error exceeds 1%:', tau_end1)

# define initial and final times 
t_start = tau_start * r0**2 / (12 * 1e11)
t_end = tau_end_i * r0**2 / (12 * 1e11)
dt = cfl_timestep(grid, 1e11)

# define initial ring with the ring solution, evolve solution, find analytic solution
sigma0 = ring_solution(grid.r, t_start, m_ring, r0, 1e11) # picked a random mass for the ring using chat gpt 
t, sigma, mass = evolve(sigma0, grid, viscosity, t_end - t_start, dt)
sigma_analytic = ring_solution(grid.r, t_end, m_ring, r0, 1e11)

# find the index where the error is at its max, find r with this index
max_err_index = np.argmax(np.abs(sigma[-1] - sigma_analytic)/np.max(sigma_analytic)*100)
r_err_max = grid.r[max_err_index]
print('R where error maximizes as it exceeds 1%:', r_err_max/c.AU, 'AU')
    
### move inner edge to .3 AU 

# define initial conditions 
r0 = 10*c.AU 
grid = make_grid(0.3*c.AU, 1000*c.AU, int(40*3))
viscosity = np.full(grid.n, 1e11)
m_ring = c.M_SUN

# convert tau into timesteps, then find the dt with cfl_timestep
tau_start = 0.01
tau_end = np.linspace(.1, .5, 100) 

# loop through each tau end and find the maximum error 
max_errs = [] 

for tau_end_i in tau_end: 
    
    # define initial and final times 
    t_start = tau_start * r0**2 / (12 * 1e11)
    t_end = tau_end_i * r0**2 / (12 * 1e11)
    
    dt = cfl_timestep(grid, 1e11)
    
    # define initial ring with the ring solution, evolve solution, find analytic solution
    sigma0 = ring_solution(grid.r, t_start, m_ring, r0, 1e11) # picked a random mass for the ring using chat gpt 
    t, sigma, mass = evolve(sigma0, grid, viscosity, t_end - t_start, dt)
    sigma_analytic = ring_solution(grid.r, t_end, m_ring, r0, 1e11)
    
    # find the error at every r compared to the analytic solution 
    max_err = np.max(np.abs(sigma[-1] - sigma_analytic)/np.max(sigma_analytic)*100)
    
    max_errs.append(max_err)

# plot errors vs tau 
ax.plot(tau_end, max_errs, c='green', label='inner edge at 0.3 AU')

# find where the errors exceed 1%, find tau where that index is, rerun for that tau 
first_i_gt1 = np.where(np.array(max_errs) >= 1)[0][0]
tau_end1 = tau_end[first_i_gt1]
print('tau where error exceeds 1%:', tau_end1)

# define initial and final times 
t_start = tau_start * r0**2 / (12 * 1e11)
t_end = tau_end_i * r0**2 / (12 * 1e11)
dt = cfl_timestep(grid, 1e11)

# define initial ring with the ring solution, evolve solution, find analytic solution
sigma0 = ring_solution(grid.r, t_start, m_ring, r0, 1e11) # picked a random mass for the ring using chat gpt 
t, sigma, mass = evolve(sigma0, grid, viscosity, t_end - t_start, dt)
sigma_analytic = ring_solution(grid.r, t_end, m_ring, r0, 1e11)

# find the index where the error is at its max, find r with this index
max_err_index = np.argmax(np.abs(sigma[-1] - sigma_analytic)/np.max(sigma_analytic)*100)
r_err_max = grid.r[max_err_index]
print('R where error maximizes as it exceeds 1%:', r_err_max/c.AU, 'AU')

# prettify
ax.set_yscale('log')
ax.set_xlabel(r'\tau')
ax.set_ylabel('Maximum Error (%)')
ax.axhline(1, c='red', label='1%')
ax.legend(frameon=False)

#%%
# 4. Do the Lupus disks drain viscously? 
# --------------------------------------

# part a 
# ------

# read in the survey data 
lupus_data = pd.read_csv('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/Electives/Origin Evolution of Planetary Systems/astr5820/data/ps3/lupus_disks.csv')

# # calculate M_dust with the equation from 2b
T_dust = 20 
distance = lupus_data['distance_pc']*c.PC
F_nu = lupus_data['F890_mJy']*1e-3*1e-26
kappa = 0.337
B = planck_nu(c.C_LIGHT/890e-6, T_dust)

# # compute dust mass 
M_dust = F_nu*distance**2 / (kappa*B)
M_disk = 100*M_dust/c.M_SUN

# plot M_dot against M_dust 
M_dot = (10**lupus_data['log_mdot'])

fig, ax = plt.subplots(1, 1)
ax.scatter(M_disk, M_dot, s=10, c='k', marker='*')

# calculate and plot the upper limit for M_dust 
F_nu_upper_indices = np.where(lupus_data['F890_limit'] == 1)[0]

ax.scatter(M_disk[F_nu_upper_indices], M_dot[F_nu_upper_indices], c='red', marker='<', label=r'upper limit on $M_{disk}$')

# make a list of flags 
flags = ['transitional', 'subluminous', 'chromospheric;transitional;mdot_upper_limit', 'subluminous;transitional', 'chromospheric;mdot_upper_limit']
flag_c = ['olivedrab', 'green', 'olive', 'yellowgreen', 'limegreen']

# loop through where these flags show up and make a scatter plot 
for f, fc in zip(flags, flag_c): 
    index = np.where(lupus_data['flag'] == f)[0]
    plt.scatter(M_disk[index], M_dot[index], c=fc, marker='*', label=f)

# prettify 
ax.set_xlabel(r'$M_{disk}$ (solar masses)')
ax.set_ylabel(r'$\dot{M}$ (solar masses/yr)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(frameon=False, bbox_to_anchor=[0, 0, .9, -.15])

# part b 
# ------

# wrote the calculation for M/M_dot in latex 

# isolate the clean sample and calculate M_disk/M_dot 
clean_indices = np.where(pd.isna(lupus_data['flag']))[0]

t_acc_yr = np.median(M_disk[clean_indices]/M_dot[clean_indices])

# convert to Myr 
print(f'Median of Mdisk/Mdot: {t_acc_yr:.1e} yr')

# find the analytical solution -- wrote in latex (used chat for help)
t_disk_yr = t_acc_yr/2 # assuming gamma is 1 

print(f'Disk Age: {t_disk_yr*1e-6:.1f} Myr')
# this limit is smaller 


# part c 
# ------

# used the jupyter notebook in 5 for this! 

def mdot_faces(sigma, grid, nu):
    q = nu * sigma * grid.r**(.5)
    q_pad = pad_with_boundaries(q)
    
    dqdr = np.diff(q_pad)/np.diff(grid.r_pad)
    m_dot = 6 * np.pi * grid.r_face**(.5) * dqdr
    return m_dot 

alpha, r1, M0d = 1e-3, 30 * c.AU, 0.05 * c.M_SUN
gd = make_grid(0.1*c.AU, 1e4*c.AU, 20*5)
nud = alpha_viscosity(gd.r, alpha, temperature)
nu1 = alpha_viscosity(r1, alpha, temperature)
t_s = r1**2 / (3 * nu1)

dt = cfl_timestep(gd, nud)
sigma0 = similarity_solution(gd.r, 0, M0d, r1, nu1, gamma=1.0)

t_d, sig_d, mass_d = evolve(sigma0, gd, nud, 10*c.MYR, dt, n_out=100)

mdot_star = np.array([mdot_faces(S_, gd, nud)[0] for S_ in sig_d])
print(f"t_s = {t_s / c.MYR:.2f} Myr")

# compare m(t) and mdot(t) 
t_tilde = 1 + t_d/t_s
M_analytical = M0d*t_tilde**(-1/2)/c.M_SUN
Mdot_analytical = 3*nu1*M0d/(2*r1**2) * t_tilde**(-3/2)*c.YR/c.M_SUN

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(t_d/c.MYR, M_analytical, c='green', label='analytical solution')
ax[1].plot(t_d/c.MYR, Mdot_analytical, c='green', label='analytical solution')
ax[0].plot(t_d/c.MYR, mass_d/c.M_SUN, c='k', linestyle='--', label='numerical solution')
ax[1].plot(t_d/c.MYR, mdot_star*c.YR/c.M_SUN, c='k', linestyle='--', label='numerical solution')

# prettify 
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[0].set_xlabel(r'time (Myr)')
ax[1].set_xlabel(r'time (Myr)')
ax[0].set_ylabel(r'$M_{disk}$ (solar masses)')
ax[1].set_ylabel(r'$\dot{M}$ (solar masses/yr)')
ax[0].legend(frameon=False)
ax[1].legend(frameon=False)

# plot our trend 
fig, ax = plt.subplots(1, 1)
ax.plot(mass_d/c.M_SUN, mdot_star*c.YR/c.M_SUN, c='k', linestyle='--', label='numerical solution')
ax.plot(M_analytical, Mdot_analytical, c='green', label='numerical solution')

# plot what corresponds to 1, 2, and 3 Myr 
i_1Myr = np.argmin(np.abs(t_d/c.MYR - 1))
i_2Myr = np.argmin(np.abs(t_d/c.MYR - 2))
i_3Myr = np.argmin(np.abs(t_d/c.MYR - 3))

# 1 myr
ax.scatter(mass_d[i_1Myr]/c.M_SUN, mdot_star[i_1Myr]*c.YR/c.M_SUN, c='lightcoral', s=50, marker='$1$', label='numerical 1Myr', zorder=10)

# 2 myr
ax.scatter(mass_d[i_2Myr]/c.M_SUN, mdot_star[i_2Myr]*c.YR/c.M_SUN, c='lightcoral', s=50, marker='$2$', label='numerical 2Myr', zorder=10)

# 3 myr
ax.scatter(mass_d[i_3Myr]/c.M_SUN, mdot_star[i_3Myr]*c.YR/c.M_SUN, c='lightcoral', s=50, marker='$3$', label='numerical 3Myr', zorder=10)


# replot 4a 
ax.scatter(M_disk, M_dot, s=10, c='k', marker='*')
ax.scatter(M_disk[F_nu_upper_indices], M_dot[F_nu_upper_indices], c='red', marker='<', label=r'upper limit on $M_{disk}$')
for f, fc in zip(flags, flag_c): 
    index = np.where(lupus_data['flag'] == f)[0]
    plt.scatter(M_disk[index], M_dot[index], c=fc, marker='*', label=f)
    
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$M_{disk}$ (solar masses)')
ax.set_ylabel(r'$\dot{M}$ (solar masses/yr)')








