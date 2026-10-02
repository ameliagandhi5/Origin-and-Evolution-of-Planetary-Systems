#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:14:04 2026

@author: ameliagandhi
"""

# imports 
import numpy as np 
import matplotlib.pyplot as plt 
from astrotools.nbody.twobody import *
from astrotools.nbody.integrators import *
from astrotools import constants as c
import pandas as pd 
from scipy.optimize import curve_fit

# define fit 
def power_fit(x, a, b):
    fit = a*x**b
    return fit 

#%%

# 1: BUILD THE MODULE 

# calculate the Hill Radius for Earth 
earth_hill_r = hill_radius(c.AU, c.M_EARTH, c.M_SUN)
print(f"Earth's Hill Radius: {earth_hill_r:.2e} m")

# calculate the roche density 
Saturn_roche_p = roche_density(1.855e8, 5.6834e26)
print(f"Roche density at Mimas' orbit: {Saturn_roche_p:.0f} kg m^-3")

# calculate the orbit position error after RK4 with the timestep and timestep halved 
# define stuff
a, e = c.AU, 0.3
P = orbital_period(a)
y0 = initial_conditions(a, e)

# The exact ellipse, for comparison. (Paul's code)
nu = np.linspace(0, 2 * np.pi, 400)
r_exact = a * (1 - e ** 2) / (1 + e * np.cos(nu))
xy_exact = np.array([r_exact * np.cos(nu), r_exact * np.sin(nu)]) / c.AU

t, y = integrate(rk4_step, kepler_derivs, y0, (0.0, P), P / 1000)
t_half, y_half = integrate(rk4_step, kepler_derivs, y0, (0.0, P), P / 2000)

fig, ax = plt.subplots(1, 1)
ax.plot(*xy_exact, 'k--', lw=1, label='exact')
ax.plot(y[:, 0] / c.AU, y[:, 1] / c.AU, '-', lw=.8, label='Euler')

err = np.hypot(*(y[-1, :2] - y0[:2]))
err_half = np.hypot(*(y_half[-1, :2] - y0[:2]))

print(err)
print(err_half)
print(f'Factor error changes after halving the timestep: {err/err_half}')

# note: increasing the step-size past this give a smaller factor after halving the timestep
# because the number is so accurate at these timesteps that the error increases by a small amount 

#%%

# 2: REFERENCE ORBIT 

# part a 
# define constants
a = c.AU 
e = 0.3 
M_star = c.M_SUN
mu = c.G*(c.M_EARTH + M_star)

# calculate aphelion and perihelion 
r_p = a*(1-e) 
r_a = a*(1+e)

# calculate orbital period and angular momentum  
P = orbital_period(a, gm=c.G*M_star)
P_yr = P/(60*60*24*365)
j = np.sqrt(mu*a*(1-e**2))

# calculate the orbital velocity at perihelion and aphelion 
v_p = np.sqrt(mu*(2/r_p - 1/a))
v_a = np.sqrt(mu*(2/r_a - 1/a))

# calculate E = -mu/2a 
E = -mu/(2*a)

print(f'r_p: {r_p/1000:.0f} km')
print(f'r_p: {r_a/1000:.0f} km')

print(f'v_p: {v_p/1000:.2f} km/s')
print(f'v_a: {v_a/1000:.2f} km/s')

print(f'E: {E:.2f} J')

print('v_p r_p:', v_p*r_p)
print('v_a r_a:', v_a*r_a)
print('j:', j)

# part b 
P = orbital_period(a)
y0 = initial_conditions(a, e)

# exact orbit 
nu = np.linspace(0, 2 * np.pi, 400)
r_exact = a * (1 - e ** 2) / (1 + e * np.cos(nu))
xy_exact = np.array([r_exact * np.cos(nu), r_exact * np.sin(nu)]) / c.AU

# RK4 orbit 
t_p2, y_p2 = integrate(rk4_step, kepler_derivs, y0, (0.0, P), P / 1000)

# errors
frac_err_p2 = np.hypot(*(y[-1, :2] - y0[:2]))/ y0[0] # position 

# specific energy
E_p2 = (specific_energy(y_p2[-1]) - specific_energy(y0))/specific_energy(y0)

# fractional error
print(f'Error in position: {frac_err_p2:.2e}')
print(f'Error in specific energy: {E_p2:.2e}')

# compare j of this orbit to j_max and j_core 

j_max = 3e14 # m^2 s^-1
j_core = 6.3e16 # m^2 s^-1

print(f'j/jmax: {j/j_max:.5f}') 
print(f'j/jcore: {j/j_core:.5f}') 


#%%

# 3: How long can an orbit calculation be trusted? 
# part a 

# redefine initial conditions 
y0 = initial_conditions(a, .3)

print('EULER')
# print(f"{'steps/orbit':>12}  {'|r - r0| / a':>14}")

fig, ax = plt.subplots(1, 1)

# loop through the errors for Euler
err_list = [] 
for n in np.logspace(0, 5, 50):
    t, y = integrate(euler_step, kepler_derivs, y0, (0.0, P), P / n, store_every=n)
    err = np.hypot(*(y[-1, :2] - y0[:2])) / a
    err_list.append(err)
    
# plot everything
ax.scatter(1/np.logspace(0, 5, 50), err_list, c='green', s=5, label='Euler')

# find fit and report slope
# use the points in the asymptotic regime 
x_fit = 1/np.logspace(0, 5, 50)[10:40]
err_for_fit = err_list[10:40]
popt_e3, pcov = curve_fit(power_fit, x_fit, err_for_fit, p0=[1,1])
print(f'Fitted slope (Euler): {popt_e3[-1]}')

print('RK4')
# print(f"{'steps/orbit':>12}  {'|r - r0| / a':>14}")

# loop through the errors for RK4 
err_list = []
for n in np.logspace(0, 5, 50):
    t, y = integrate(rk4_step, kepler_derivs, y0, (0.0, P), P / n, store_every=n)
    err = np.hypot(*(y[-1, :2] - y0[:2])) / a
    err_list.append(err)
    
# plot everything
ax.scatter(1/np.logspace(0, 5, 50), err_list, c='blue', s=5, label='RK4')

# find fit and report slope
err_for_fit = err_list[10:40]
popt_rk43, pcov = curve_fit(power_fit, x_fit, err_for_fit, p0=[1,1])
print(f'Fitted slope (RK4): {popt_rk43[-1]}')

# plot the nice things 
ax.set_xscale('log')
ax.set_yscale('log')

# slopes 
x = np.logspace(-5, 0, 50)
ax.plot(x, 1000*x, c='k', label='$y=x$')
ax.plot(x,1000*x**4, c='gray', label=r'$y=x^4$')

ax.plot(x_fit, popt_e3[0]*x_fit**popt_e3[1], c='yellowgreen')
ax.plot(x_fit, popt_rk43[0]*x_fit**popt_rk43[1], c='skyblue')

ax.legend(frameon=False)
ax.set_xlabel(r'$\Delta t/P$')
ax.set_ylabel('Error in position (a)')

#%%
# part b (also used for part e)
# calculate drift for eccentricity = 0.3
y0 = initial_conditions(a, .3)
t_e, y_e = integrate(euler_step, kepler_derivs, y0, (0.0, 4000*P), P/500, store_every=500)
t_rk4, y_rk4 = integrate(rk4_step, kepler_derivs, y0, (0.0, 4000*P), P/500, store_every=500)
drift_e = relative_drift(specific_energy(y_e))
drift_rk4 = relative_drift(specific_energy(y_rk4))

# find the average drift per orbit 
dpo_e = np.average(np.diff(drift_e))
dpo_rk4 = np.average(np.diff(drift_rk4))

print('e=.3')
print(f"Average drift per orbit Euler {dpo_e:+.5f}")
print(f"Average drift per orbit RK4 {dpo_rk4:+.3e}")

# plot 
plt.figure(figsize=(6, 5))
plt.plot(t_e / P, np.abs(drift_e), c='green', label='Euler')
plt.plot(t_rk4 / P, np.abs(drift_rk4),  c='blue', label='RK4')

# plot nice things 
plt.xlabel('Time (Orbits)')
plt.ylabel(r'$\Delta E / |E_0|$')
plt.title(r'Euler and RK4 at $\Delta t = P/500$')
plt.axhline(1e-6)
plt.tight_layout()
plt.yscale('log')
plt.legend(frameon=False)

# find the orbit where we approach 1e-6
orbit_e = t_rk4[np.argmin(np.abs(np.abs(drift_rk4) - 1e-6))]/P

#%%
# part c 

# redefine initial conditions 
y0 = initial_conditions(a, .9)

print('EULER')
# print(f"{'steps/orbit':>12}  {'|r - r0| / a':>14}")

fig, ax = plt.subplots(1, 1)

# loop through the errors for Euler
err_list = [] 
for n in np.logspace(0, 5, 50):
    t, y = integrate(euler_step, kepler_derivs, y0, (0.0, P), P / n, store_every=n)
    err = np.hypot(*(y[-1, :2] - y0[:2])) / a
    err_list.append(err)
    
# plot everything
ax.scatter(1/np.logspace(0, 5, 50), err_list, c='green', s=5, label='Euler')

# find fit and report slope
x_fit = 1/np.logspace(0, 5, 50)[30:50]
err_for_fit = err_list[30:50]
popt_e9, pcov = curve_fit(power_fit, x_fit, err_for_fit, p0=[1,1])
print(f'Fitted slope (Euler): {popt_e9[-1]}')

print('RK4')
# print(f"{'steps/orbit':>12}  {'|r - r0| / a':>14}")

# loop through the errors for RK4 
err_list = []
for n in np.logspace(0, 5, 50):
    t, y = integrate(rk4_step, kepler_derivs, y0, (0.0, P), P / n, store_every=n)
    err = np.hypot(*(y[-1, :2] - y0[:2])) / a
    err_list.append(err)
    
# plot everything
ax.scatter(1/np.logspace(0, 5, 50), err_list, c='blue', s=5, label='RK4')

# find fit and report slope
err_for_fit = err_list[30:50]
popt_rk49, pcov = curve_fit(power_fit, x_fit, err_for_fit, p0=[1,1])
print(f'Fitted slope (RK4): {popt_rk49[-1]}')

# plot the nice things 
ax.set_xscale('log')
ax.set_yscale('log')

# slopes 
x = np.logspace(-5, 0, 50)
ax.plot(x, 1000*x, c='k', label='$y=x$')
ax.plot(x,1000*x**4, c='gray', label=r'$y=x^4$')

ax.plot(x_fit, popt_e9[0]*x_fit**popt_e9[1], c='yellowgreen')
ax.plot(x_fit, popt_rk49[0]*x_fit**popt_rk49[1], c='skyblue')

ax.legend(frameon=False)
ax.set_xlabel(r'$\Delta t/P$')
ax.set_ylabel('Error in position (a)')

# compare the fits of the RK4 method at different eccentricities with the fit 
print('fit for e=.3', f'{popt_rk43[0]:.2e}', popt_rk43[1])
print('fit for e=.9', f'{popt_rk49[0]:.2e}', popt_rk49[1])
ratio_fit = popt_rk49[0]/popt_rk43[0]
print(ratio_fit)

#%%
# 4: The Hill radius and the Edge of a Satellite System

# read in the dataframe 
planets_csv = '/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/Electives/Origin Evolution of Planetary Systems/astr5820/data/ps2/planet_elements.csv'
planets_df = pd.read_csv(planets_csv)

# calculate the hill radius for each planet and make a latex table 
hill_r_planets = hill_radius(planets_df['a_au']*c.AU, planets_df['mass_kg'], c.M_SUN)

print(f'Hill Radius of Earth: {hill_r_planets[2]:.2e} m')

# make a latex table 
hill_r_planets_table = pd.DataFrame({
    'Planet': planets_df['name'],
    'Hill Radius (m)': hill_r_planets,
    'Hill Radius (AU)': hill_r_planets/c.AU})

hill_r_planets_table['Hill Radius (m)'] = hill_r_planets_table['Hill Radius (m)'].map(lambda x: f'{x:.2e}')
hill_r_planets_table['Hill Radius (AU)'] = hill_r_planets_table['Hill Radius (AU)'].map(lambda x: f'{x:.3f}')
hill_r_planets_table.to_latex('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/Electives/Origin Evolution of Planetary Systems/hillr.tex',index=False,escape=False, longtable=True)

#%%
# read in the satellites dataframe 
sats_csv = '/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/Electives/Origin Evolution of Planetary Systems/astr5820/data/ps2/satellites.csv'
sats_df = pd.read_csv(sats_csv)

prograde_sats = sats_df[sats_df['direction'] == 'prograde']
retrograde_sats = sats_df[sats_df['direction'] == 'retrograde']

for nam in ['Jupiter', 'Saturn', 'Uranus', 'Neptune']:
    planet_sats_p = prograde_sats[prograde_sats['planet'] == nam]
    planet_sats_r = retrograde_sats[retrograde_sats['planet'] == nam]
    
    hill_r_planet = float(hill_r_planets_table[hill_r_planets_table['Planet'] == nam]['Hill Radius (m)'].values[0])

    p_sat_table = pd.DataFrame({
        'Satellite': planet_sats_p['name'],
        'Semi-Major Axis': np.array(planet_sats_p['a_m'])/hill_r_planet})
    
    r_sat_table = pd.DataFrame({
        'Satellite': planet_sats_r['name'],
        'Semi-Major Axis': np.array(planet_sats_r['a_m'])/hill_r_planet})
    
    p_sat_table['Semi-Major Axis'] = p_sat_table['Semi-Major Axis'].map(lambda x: f'{x:.3f}')
    r_sat_table['Semi-Major Axis'] = r_sat_table['Semi-Major Axis'].map(lambda x: f'{x:.3f}')
    
    p_sat_table.to_latex(f'/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/Electives/Origin Evolution of Planetary Systems/hillr_sat_{nam}_p.tex',index=False,escape=False, longtable=True)
    r_sat_table.to_latex(f'/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/Electives/Origin Evolution of Planetary Systems/hillr_sat_{nam}_r.tex',index=False,escape=False, longtable=True)


    max_p_planet = planet_sats_p.iloc[np.argmax((planet_sats_p['a_m'])/hill_r_planet)]
    max_nam_p = max_p_planet['name']
    max_ratio_p = np.max((planet_sats_p['a_m'])/hill_r_planet)
    
    max_r_planet = planet_sats_r.iloc[np.argmax((planet_sats_r['a_m'])/hill_r_planet)]
    max_nam_r = max_r_planet['name']
    max_ratio_r = np.max((planet_sats_r['a_m'])/hill_r_planet)
    
    print(nam)
    print(f'Maximum a/R_H (prograde): {max_nam_p}, {max_ratio_p:.2f}')
    print(f'Maximum a/R_H (retrograde): {max_nam_r}, {max_ratio_r:.2f}')
    
    
#%%
# d

# define saturn's mass, radius, density, and the distance to the outer ring 
m_saturn = 5.68e26 # kg
r_saturn = 5.43e7 # m
rho_saturn = 687 # kg m-3
dist_ring = 1.368e8 # m

# find the density for a rigid body in saturn's ring 
rho_rigid = rho_saturn*(1.44*r_saturn/dist_ring)**3

# find the density for a fluid body in saturn's ring 
rho_fluid = rho_saturn*(2.44*r_saturn/dist_ring)**3

print(f'rigid model density: {rho_rigid} kgm-3')
print(f'fluid model density: {rho_fluid} kgm-3')
