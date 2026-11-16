#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:03:42 2026

@author: ameliagandhi
"""

# boilerplate 
import numpy as np 
import scipy 
import matplotlib.pyplot as plt 
from astrotools.nbody.integrators import * 
from astrotools.nbody.drag import * 
import astrotools.constants as c
import sys 
import rebound

#%%

# 1: Build the Module 
# -------------------

### far from the planet, a pebble keeps its drift velocity 
y0 = initial_state(1, 0.1, 0.41)

vx, vy, ax, ay = hill_drag_derivs(1, y0, 0.1, 0.41)

print(f'|ax|: {np.abs(ax):.5f}')
print(f'|ay + 3/2 vx|: {np.abs(ay + 3/2*vx):.5f}') 
print(f'3/r^2: {3/(y0[0]**2 + y0[1]**2):.5f}')


### with encounter and certain conditions, a pebble starting at x0 = 1 is acptured and one starting at x=2 is not 
hill_drag_func = lambda t, y: hill_drag_derivs(t, y, 0.1, 0.41)

# determine if particle at x=1 is captured (can use y0 from first test)
captured, t, states = encounter(rk4_step, hill_drag_func, y0, 0.05, t_max=None, dt_max=0.02)
print(f'particle at x0=1 is captured: {captured}')

# change x0=2 with initial conditions 
y0 = initial_state(2, 0.1, 0.41) 

# determine if particle at x=1 is captured (can use y0 from first test)
captured, t, states = encounter(rk4_step, hill_drag_func, y0, 0.05, t_max=None, dt_max=0.02)
print(f'particle at x0=2 is captured: {captured}')

#%%

# 2: Capturing pebbles 
# --------------------

### part a 

# define variables 
dv = 0.41; st = 0.1; r_cap = 0.05; delx0 = 0.05

# define the hill drag function 
hill_drag_func = lambda t, y: hill_drag_derivs(t, y, st, dv)

# define width x0 = 10 rH
x0_range = np.arange(0, 6, delx0)

# define the initial conditions with x0_range 
y0_list = [initial_state(x0, 0.1, 0.41) for x0 in x0_range]

# make a figure to plot stuff on 
fig, ax = plt.subplots(1,1, figsize=(10, 6))

# loop through the initial conditions and determine if captured or not with encounter 
# plot all trajectories and color code based on if captured or not 
# store stuff for captured objects only in the list below 
x0_captured_list = [] 
vy0_captured_list = [] 

for y0 in y0_list: 
    captured, t, states = encounter(rk4_step, hill_drag_func, y0, r_cap, t_max=None, dt_max=0.02)
    if captured: 
        ax.plot(states[:,0], states[:,1], c='yellowgreen')
        
        x0_captured_list.append(y0[0])
        vy0_captured_list.append(y0[3])
        
    else: 
        ax.plot(states[:,0], states[:,1], c='gray')

# calculate accretion rate numerically 
Mdot_numeric = np.sum(np.abs(np.array(vy0_captured_list))*delx0)

# calculate accretion rate analytically 
Mdot_analytical = 3*(3*st)**(2/3) 

print(f'M_dot \n numeric: {Mdot_numeric:.2f} \n analytical: {Mdot_analytical:.2f}')
        
# prettify 
ax.scatter(0, 0, s=30, c='k', zorder=10)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_xlabel(r'x ($r_H$)')
ax.set_ylabel(r'y ($r_H$)')
ax.set_title(rf'$x_0$ range: {np.min(np.array(x0_captured_list)):.2f}-{np.max(np.array(x0_captured_list)):.2f} $r_H$') 

#%%

### part b

# define variables 
dv = 0.41; st = 1; r_cap = 0.05; delx0 = 0.05

# define the hill drag function 
hill_drag_func = lambda t, y: hill_drag_derivs(t, y, st, dv)

# define width x0 = 10 rH
x0_range = np.arange(0, 6, delx0)

# define the initial conditions with x0_range 
y0_list = [initial_state(x0, 0.1, 0.41) for x0 in x0_range]

# make a figure to plot stuff on 
fig, ax = plt.subplots(1,1, figsize=(10, 6))

# loop through the initial conditions and determine if captured or not with encounter 
# plot all trajectories and color code based on if captured or not 
# store stuff for captured objects only in the list below 
x0_captured_list = [] 
vy0_captured_list = [] 

for y0 in y0_list: 
    captured, t, states = encounter(rk4_step, hill_drag_func, y0, r_cap, t_max=None, dt_max=0.02)
    if captured: 
        ax.plot(states[:,0], states[:,1], c='yellowgreen')
        
        x0_captured_list.append(y0[0])
        vy0_captured_list.append(y0[3])
        
    else: 
        ax.plot(states[:,0], states[:,1], c='gray')

# calculate accretion rate numerically 
Mdot_numeric = np.sum(np.abs(np.array(vy0_captured_list))*delx0)

# calculate accretion rate analytically 
Mdot_analytical = 3*(3*st)**(2/3) 

print(f'M_dot \n numeric: {Mdot_numeric:.2f} \n analytical: {Mdot_analytical:.2f}')
        
# prettify 
ax.scatter(0, 0, s=30, c='k', zorder=10)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_xlabel(r'x ($r_H$)')
ax.set_ylabel(r'y ($r_H$)')
ax.set_title(rf'$x_0$ range: {np.min(np.array(x0_captured_list)):.2f}-{np.max(np.array(x0_captured_list)):.2f} $r_H$') 


#%%

### part c

# define variables 
dv = 0.41; st = 0.1; r_cap = 0.01; delx0 = 0.05

# define the hill drag function 
hill_drag_func = lambda t, y: hill_drag_derivs(t, y, st, dv)

# define width x0 = 10 rH
x0_range = np.arange(0, 6, delx0)

# define the initial conditions with x0_range 
y0_list = [initial_state(x0, 0.1, 0.41) for x0 in x0_range]

# make a figure to plot stuff on 
fig, ax = plt.subplots(1,1, figsize=(10, 6))

# loop through the initial conditions and determine if captured or not with encounter 
# plot all trajectories and color code based on if captured or not 
# store stuff for captured objects only in the list below 
x0_captured_list = [] 
vy0_captured_list = [] 

for y0 in y0_list: 
    captured, t, states = encounter(rk4_step, hill_drag_func, y0, r_cap, t_max=None, dt_max=0.02)
    if captured: 
        ax.plot(states[:,0], states[:,1], c='yellowgreen')
        
        x0_captured_list.append(y0[0])
        vy0_captured_list.append(y0[3])
        
    else: 
        ax.plot(states[:,0], states[:,1], c='gray')

# find range of x0 where the pebbles are captured 

# calculate accretion rate numerically 
Mdot_numeric = np.sum(np.abs(np.array(vy0_captured_list))*delx0)

# calculate accretion rate analytically 
Mdot_analytical = 3*(3*st)**(2/3) 

print(f'M_dot \n numeric: {Mdot_numeric:.2f} \n analytical: {Mdot_analytical:.2f}')
        
# prettify 
ax.scatter(0, 0, s=30, c='k', zorder=10)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_xlabel(r'x ($r_H$)')
ax.set_ylabel(r'y ($r_H$)')
ax.set_title(rf'$x_0$ range: {np.min(np.array(x0_captured_list)):.2f}-{np.max(np.array(x0_captured_list)):.2f} $r_H$') 

## remove the gas and recompute: rcap = 0.05

# define variables 
dv = 0; st = 1e6; r_cap = 0.05; delx0 = 0.02

# define the hill drag function 
hill_drag_func = lambda t, y: hill_drag_derivs(t, y, st, dv)

# define width x0 = 10 rH
x0_range = np.arange(-3, 3, delx0)

# define the initial conditions with x0_range 
y0_list = [initial_state(x0, 0.1, 0) for x0 in x0_range]

# make a figure to plot stuff on 
fig, ax = plt.subplots(1,1, figsize=(10, 6))

# loop through the initial conditions and determine if captured or not with encounter 
# plot all trajectories and color code based on if captured or not 
# store stuff for captured objects only in the list below 
x0_captured_list = [] 
vy0_captured_list = [] 

for y0 in y0_list: 
    captured, t, states = encounter(rk4_step, hill_drag_func, y0, r_cap, t_max=None, dt_max=0.02)
    if captured: 
        ax.plot(states[:,0], states[:,1], c='yellowgreen')
        
        x0_captured_list.append(y0[0])
        vy0_captured_list.append(y0[3])
        
    else: 
        ax.plot(states[:,0], states[:,1], c='gray')

# find range of x0 where the pebbles are captured 

# calculate accretion rate numerically 
Mdot_numeric = np.sum(np.abs(np.array(vy0_captured_list))*delx0)

# calculate accretion rate analytically 
Mdot_analytical = 3*(3*st)**(2/3) 

print(f'M_dot \n numeric: {Mdot_numeric:.2f} \n analytical: {Mdot_analytical:.2f}')
        
# prettify 
ax.scatter(0, 0, s=30, c='k', zorder=10)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_xlabel(r'x ($r_H$)')
ax.set_ylabel(r'y ($r_H$)')
ax.set_title(rf'$x_0$ range: {np.min(np.array(x0_captured_list)):.2f}-{np.max(np.array(x0_captured_list)):.2f} $r_H$') 

## remove the gas and recompute: rcap = 0.01

# define variables 
dv = 0; st = 1e6; r_cap = 0.01; delx0 = 0.02

# define the hill drag function 
hill_drag_func = lambda t, y: hill_drag_derivs(t, y, st, dv)

# define width x0 = 10 rH
x0_range = np.arange(-3, 3, delx0)

# define the initial conditions with x0_range 
y0_list = [initial_state(x0, 0.1, 0) for x0 in x0_range]

# make a figure to plot stuff on 
fig, ax = plt.subplots(1,1, figsize=(10, 6))

# loop through the initial conditions and determine if captured or not with encounter 
# plot all trajectories and color code based on if captured or not 
# store stuff for captured objects only in the list below 
x0_captured_list = [] 
vy0_captured_list = [] 

for y0 in y0_list: 
    captured, t, states = encounter(rk4_step, hill_drag_func, y0, r_cap, t_max=None, dt_max=0.02)
    if captured: 
        ax.plot(states[:,0], states[:,1], c='yellowgreen')
        
        x0_captured_list.append(y0[0])
        vy0_captured_list.append(y0[3])
        
    else: 
        ax.plot(states[:,0], states[:,1], c='gray')

# find range of x0 where the pebbles are captured 

# calculate accretion rate numerically 
Mdot_numeric = np.sum(np.abs(np.array(vy0_captured_list))*delx0)

# calculate accretion rate analytically 
Mdot_analytical = 3*(3*st)**(2/3) 

print(f'M_dot \n numeric: {Mdot_numeric:.2f} \n analytical: {Mdot_analytical:.2f}')
        
# prettify 
ax.scatter(0, 0, s=30, c='k', zorder=10)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_xlabel(r'x ($r_H$)')
ax.set_ylabel(r'y ($r_H$)')
ax.set_title(rf'$x_0$ range: {np.min(np.array(x0_captured_list)):.2f}-{np.max(np.array(x0_captured_list)):.2f} $r_H$') 

#%%

# 3: Holding Trojans
# ------------------

### part a 

### NOTE: I wanted to run 30 orbits too (plots look nicer), so the code is below this block 

### simulation setup 

# make simulation and units 
sim = rebound.Simulation()
sim.units = ("yr", "AU", "Msun")

# define variables 
a = 5.2
e = 1e-3
mu = 9.54e-4
Omega = np.sqrt(sim.G/a**3)
L4x = a*(.5-mu)
L4y = np.sqrt(3)/2*a

# add particles 
# used AI to figure out what vx and vy are 
sim.add(m=1-mu, x=-mu*a, y=0,  vy=-mu*a*Omega) # Star
sim.add(m=mu, x=(1-mu)*a, y=0, vy=(1-mu)*a*Omega) # planet 
sim.add(m=0, x=e*a + L4x, y=e*a + L4y, vx=-Omega*(e*a + L4y), vy=Omega*(e*a + L4x))  # massless object
sim.move_to_com() # move to center of mass 

# get the period (yrs)
period = sim.particles[1].orbit(primary=sim.particles[0]).P

### Integrate with IAS15 for 300 planetary orbits
sim.integrator = "ias15"

# define times, xy coordinates 
times = np.linspace(0, period*300, 3000) 
xy = np.empty((len(times), 3, 2))

for k, t in enumerate(times):
    sim.integrate(t)
    for i, p in enumerate(sim.particles):
        xy[k, i] = p.x, p.y

fig, ax = plt.subplots(figsize=(5, 5))
for i, (name, c) in enumerate([("Sun","k"), ("Jupiter","C1"), ("Test Particle","C2")]):
    ax.plot(xy[:, i, 0], xy[:, i, 1], color=c, lw=0.8, label=name)
    ax.set(aspect="equal", xlabel="x (AU)", ylabel="y (AU)", title="300 Jovian Orbits, center-of-mass frame")
    ax.legend()

### measure the angle between the particle and the planet 

# define a function 
def angle_from_star(xy, i, j):
    """Angle of body j relative to body i as seen from body 0, in degrees."""
    d_i = xy[:, i] - xy[:, 0]
    d_j = xy[:, j] - xy[:, 0]
    ang = np.arctan2(d_j[:, 1], d_j[:, 0]) - np.arctan2(d_i[:, 1], d_i[:, 0])
    return np.degrees((ang + np.pi) % (2 * np.pi) - np.pi)

phi = angle_from_star(xy, 1, 2)

# compute the range of phi
print(f'Range of phi: {np.min(phi):.2f} - {np.max(phi):.2f}')

# numerical libration period 
cross = np.where((phi[:-1] > np.average(phi)) & (phi[1:] <= np.average(phi)))[0] # find where we cross the mean phi
libration_P_numeric = np.mean(np.diff(times[cross]))
P_J, P_test = (sim.particles[i].orbit(primary=sim.particles[0]).P for i in (1, 2))

# analytic period 
libration_P_analytic = P_J/np.sqrt(27*mu/4)

# compute the percent error 
p_err_libration_P = np.abs(libration_P_numeric - libration_P_analytic)/libration_P_analytic*100

print(f"measured synodic period {libration_P_numeric:.1f} yr; "f"predicted {libration_P_analytic:.1f} yr")
print(f'Libration period percent error: {p_err_libration_P:.3f}%')

# plot phi(t) 
fig_l, ax_l = plt.subplots(figsize=(8, 4))
ax_l.plot(times, phi, lw=0.9, label=r'$\mu = 9.54 \times 10^{-4}$')

### part b 

## mu = 0.05
# make simulation and units 
sim = rebound.Simulation()
sim.units = ("yr", "AU", "Msun")

# define variables 
a = 5.2
e = 1e-3
mu = 0.03
Omega = np.sqrt(sim.G/a**3)
L4x = a*(.5-mu)
L4y = np.sqrt(3)/2*a

# add particles 
# used AI to figure out what vx and vy are 
sim.add(m=1-mu, x=-mu*a, y=0,  vy=-mu*a*Omega) # Star
sim.add(m=mu, x=(1-mu)*a, y=0, vy=(1-mu)*a*Omega) # planet 
sim.add(m=0, x=e*a + L4x, y=e*a + L4y, vx=-Omega*(e*a + L4y), vy=Omega*(e*a + L4x))  # massless object
sim.move_to_com() # move to center of mass 

# get the period (yrs)
period = sim.particles[1].orbit(primary=sim.particles[0]).P

### Integrate with IAS15 for 300 planetary orbits
sim.integrator = "ias15"

# define times, xy coordinates 
times = np.linspace(0, period*300, 3000) 
xy = np.empty((len(times), 3, 2))

for k, t in enumerate(times):
    sim.integrate(t)
    for i, p in enumerate(sim.particles):
        xy[k, i] = p.x, p.y

fig, ax = plt.subplots(figsize=(5, 5))
for i, (name, c) in enumerate([("Sun","k"), ("Jupiter","C1"), ("Test Particle","C2")]):
    ax.plot(xy[:, i, 0], xy[:, i, 1], color=c, lw=0.8, label=name)
    ax.set(aspect="equal", xlabel="x (AU)", ylabel="y (AU)", title="300 Jovian Orbits, center-of-mass frame")
    ax.legend()

### measure the angle between the particle and the planet 

phi = angle_from_star(xy, 1, 2)

# compute the range of phi
print(f'Range of phi: {np.min(phi):.2f} - {np.max(phi):.2f}')

# numerical libration period 
cross = np.where((phi[:-1] > np.average(phi)) & (phi[1:] <= np.average(phi)))[0] # find where we cross the mean phi
libration_P_numeric = np.mean(np.diff(times[cross]))
P_J, P_test = (sim.particles[i].orbit(primary=sim.particles[0]).P for i in (1, 2))

# analytic period 
libration_P_analytic = P_J/np.sqrt(27*mu/4)

# compute the percent error 
p_err_libration_P = np.abs(libration_P_numeric - libration_P_analytic)/libration_P_analytic*100

print(f"measured synodic period {libration_P_numeric:.1f} yr; "f"predicted {libration_P_analytic:.1f} yr")
print(f'Libration period percent error: {p_err_libration_P:.3f}%')


# µ≪1, small displacements from L4 librate with period P/ 27µ/4, where P is the planet’s
# orbital period, and L4 is linearly stable for µ(1−µ) <1/27

# plot phi(t) 
ax_l.plot(times, phi, lw=0.9, label=r'$\mu = 0.03$')

## mu = 0.05
# make simulation and units 
sim = rebound.Simulation()
sim.units = ("yr", "AU", "Msun")

# define variables 
a = 5.2
e = 1e-3
mu = 0.05
Omega = np.sqrt(sim.G/a**3)
L4x = a*(.5-mu)
L4y = np.sqrt(3)/2*a

# add particles 
# used AI to figure out what vx and vy are 
sim.add(m=1-mu, x=-mu*a, y=0,  vy=-mu*a*Omega) # Star
sim.add(m=mu, x=(1-mu)*a, y=0, vy=(1-mu)*a*Omega) # planet 
sim.add(m=0, x=e*a + L4x, y=e*a + L4y, vx=-Omega*(e*a + L4y), vy=Omega*(e*a + L4x))  # massless object
sim.move_to_com() # move to center of mass 

# get the period (yrs)
period = sim.particles[1].orbit(primary=sim.particles[0]).P

### Integrate with IAS15 for 300 planetary orbits
sim.integrator = "ias15"

# define times, xy coordinates 
times = np.linspace(0, period*300, 3000) 
xy = np.empty((len(times), 3, 2))

for k, t in enumerate(times):
    sim.integrate(t)
    for i, p in enumerate(sim.particles):
        xy[k, i] = p.x, p.y

fig, ax = plt.subplots(figsize=(5, 5))
for i, (name, c) in enumerate([("Sun","k"), ("Jupiter","C1"), ("Test Particle","C2")]):
    ax.plot(xy[:, i, 0], xy[:, i, 1], color=c, lw=0.8, label=name)
    ax.set(aspect="equal", xlabel="x (AU)", ylabel="y (AU)", title="300 Jovian Orbits, center-of-mass frame")
    ax.legend()

### measure the angle between the particle and the planet 

phi = angle_from_star(xy, 1, 2)

# compute the range of phi
print(f'Range of phi: {np.min(phi):.2f} - {np.max(phi):.2f}')

# numerical libration period 
cross = np.where((phi[:-1] > np.average(phi)) & (phi[1:] <= np.average(phi)))[0] # find where we cross the mean phi
libration_P_numeric = np.mean(np.diff(times[cross]))
P_J, P_test = (sim.particles[i].orbit(primary=sim.particles[0]).P for i in (1, 2))

# analytic period 
libration_P_analytic = P_J/np.sqrt(27*mu/4)

# compute the percent error 
p_err_libration_P = np.abs(libration_P_numeric - libration_P_analytic)/libration_P_analytic*100

print(f"measured synodic period {libration_P_numeric:.1f} yr; "f"predicted {libration_P_analytic:.1f} yr")
print(f'Libration period percent error: {p_err_libration_P:.3f}%')

# plot phi(t) 
# needed to do this on a separate graph because this particle escapes (it is not stable)
fig_l2, ax_l2 = plt.subplots(figsize=(8, 4))
ax_l2.plot(times, phi, lw=0.9, label=r'$\mu = 0.05$')


# prettify plot 
ax_l.set(xlabel="time (yr)", ylabel=r"$\phi_{Particle} - \phi_{Jupiter}$ (deg)", ylim=(50, 70))
ax_l.legend(frameon=False)
ax_l2.set(xlabel="time (yr)", ylabel=r"$\phi_{Particle} - \phi_{Jupiter}$ (deg)", ylim=(-180, 180))
ax_l2.legend(frameon=False)

#%%

### part a 

### simulation setup 

# make simulation and units 
sim = rebound.Simulation()
sim.units = ("yr", "AU", "Msun")

# define variables 
a = 5.2
e = 1e-3
mu = 9.54e-4
Omega = np.sqrt(sim.G/a**3)
L4x = a*(.5-mu)
L4y = np.sqrt(3)/2*a

# add particles 
# used AI to figure out what vx and vy are 
sim.add(m=1-mu, x=-mu*a, y=0,  vy=-mu*a*Omega) # Star
sim.add(m=mu, x=(1-mu)*a, y=0, vy=(1-mu)*a*Omega) # planet 
sim.add(m=0, x=e*a + L4x, y=e*a + L4y, vx=-Omega*(e*a + L4y), vy=Omega*(e*a + L4x))  # massless object
sim.move_to_com() # move to center of mass 

# get the period (yrs)
period = sim.particles[1].orbit(primary=sim.particles[0]).P

### Integrate with IAS15 for 300 planetary orbits
sim.integrator = "ias15"

# define times, xy coordinates 
times = np.linspace(0, period*30, 3000) 
xy = np.empty((len(times), 3, 2))

for k, t in enumerate(times):
    sim.integrate(t)
    for i, p in enumerate(sim.particles):
        xy[k, i] = p.x, p.y

fig, ax = plt.subplots(figsize=(5, 5))
for i, (name, c) in enumerate([("Sun","k"), ("Jupiter","C1"), ("Test Particle","C2")]):
    ax.plot(xy[:, i, 0], xy[:, i, 1], color=c, lw=0.8, label=name)
    ax.set(aspect="equal", xlabel="x (AU)", ylabel="y (AU)", title="30 Jovian Orbits, center-of-mass frame")
    ax.legend()

### measure the angle between the particle and the planet 

phi = angle_from_star(xy, 1, 2)

# numerical libration period 
cross = np.where((phi[:-1] > np.average(phi)) & (phi[1:] <= np.average(phi)))[0] # find where we cross the mean phi
libration_P_numeric = np.mean(np.diff(times[cross]))
P_J, P_test = (sim.particles[i].orbit(primary=sim.particles[0]).P for i in (1, 2))

# analytic period 
libration_P_analytic = P_J/np.sqrt(27*mu/4)

# compute the percent error 
p_err_libration_P = np.abs(libration_P_numeric - libration_P_analytic)/libration_P_analytic*100

print(f"measured synodic period {libration_P_numeric:.1f} yr; "f"predicted {libration_P_analytic:.1f} yr")
print(f'Libration period percent error: {p_err_libration_P:.3f}%')

# plot phi(t) 
fig_l, ax_l = plt.subplots(figsize=(8, 4))
ax_l.plot(times, phi, lw=0.9, label=r'$\mu = 9.54 \times 10^{-4}$')

### part b 

## mu = 0.05
# make simulation and units 
sim = rebound.Simulation()
sim.units = ("yr", "AU", "Msun")

# define variables 
a = 5.2
e = 1e-3
mu = 0.03
Omega = np.sqrt(sim.G/a**3)
L4x = a*(.5-mu)
L4y = np.sqrt(3)/2*a

# add particles 
# used AI to figure out what vx and vy are 
sim.add(m=1-mu, x=-mu*a, y=0,  vy=-mu*a*Omega) # Star
sim.add(m=mu, x=(1-mu)*a, y=0, vy=(1-mu)*a*Omega) # planet 
sim.add(m=0, x=e*a + L4x, y=e*a + L4y, vx=-Omega*(e*a + L4y), vy=Omega*(e*a + L4x))  # massless object
sim.move_to_com() # move to center of mass 

# get the period (yrs)
period = sim.particles[1].orbit(primary=sim.particles[0]).P

### Integrate with IAS15 for 300 planetary orbits
sim.integrator = "ias15"

# define times, xy coordinates 
times = np.linspace(0, period*30, 3000) 
xy = np.empty((len(times), 3, 2))

for k, t in enumerate(times):
    sim.integrate(t)
    for i, p in enumerate(sim.particles):
        xy[k, i] = p.x, p.y

fig, ax = plt.subplots(figsize=(5, 5))
for i, (name, c) in enumerate([("Sun","k"), ("Jupiter","C1"), ("Test Particle","C2")]):
    ax.plot(xy[:, i, 0], xy[:, i, 1], color=c, lw=0.8, label=name)
    ax.set(aspect="equal", xlabel="x (AU)", ylabel="y (AU)", title="30 Jovian Orbits, center-of-mass frame")
    ax.legend()

### measure the angle between the particle and the planet 

phi = angle_from_star(xy, 1, 2)

# numerical libration period 
cross = np.where((phi[:-1] > np.average(phi)) & (phi[1:] <= np.average(phi)))[0] # find where we cross the mean phi
libration_P_numeric = np.mean(np.diff(times[cross]))
P_J, P_test = (sim.particles[i].orbit(primary=sim.particles[0]).P for i in (1, 2))

# analytic period 
libration_P_analytic = P_J/np.sqrt(27*mu/4)

# compute the percent error 
p_err_libration_P = np.abs(libration_P_numeric - libration_P_analytic)/libration_P_analytic*100

print(f"measured synodic period {libration_P_numeric:.1f} yr; "f"predicted {libration_P_analytic:.1f} yr")
print(f'Libration period percent error: {p_err_libration_P:.3f}%')


# µ≪1, small displacements from L4 librate with period P/ 27µ/4, where P is the planet’s
# orbital period, and L4 is linearly stable for µ(1−µ) <1/27

# plot phi(t) 
ax_l.plot(times, phi, lw=0.9, label=r'$\mu = 0.03$')

## mu = 0.05
# make simulation and units 
sim = rebound.Simulation()
sim.units = ("yr", "AU", "Msun")

# define variables 
a = 5.2
e = 1e-3
mu = 0.05
Omega = np.sqrt(sim.G/a**3)
L4x = a*(.5-mu)
L4y = np.sqrt(3)/2*a

# add particles 
# used AI to figure out what vx and vy are 
sim.add(m=1-mu, x=-mu*a, y=0,  vy=-mu*a*Omega) # Star
sim.add(m=mu, x=(1-mu)*a, y=0, vy=(1-mu)*a*Omega) # planet 
sim.add(m=0, x=e*a + L4x, y=e*a + L4y, vx=-Omega*(e*a + L4y), vy=Omega*(e*a + L4x))  # massless object
sim.move_to_com() # move to center of mass 

# get the period (yrs)
period = sim.particles[1].orbit(primary=sim.particles[0]).P

### Integrate with IAS15 for 300 planetary orbits
sim.integrator = "ias15"

# define times, xy coordinates 
times = np.linspace(0, period*30, 3000) 
xy = np.empty((len(times), 3, 2))

for k, t in enumerate(times):
    sim.integrate(t)
    for i, p in enumerate(sim.particles):
        xy[k, i] = p.x, p.y

fig, ax = plt.subplots(figsize=(5, 5))
for i, (name, c) in enumerate([("Sun","k"), ("Jupiter","C1"), ("Test Particle","C2")]):
    ax.plot(xy[:, i, 0], xy[:, i, 1], color=c, lw=0.8, label=name)
    ax.set(aspect="equal", xlabel="x (AU)", ylabel="y (AU)", title="30 Jovian Orbits, center-of-mass frame")
    ax.legend()

### measure the angle between the particle and the planet 

phi = angle_from_star(xy, 1, 2)

# numerical libration period 
cross = np.where((phi[:-1] > np.average(phi)) & (phi[1:] <= np.average(phi)))[0] # find where we cross the mean phi
libration_P_numeric = np.mean(np.diff(times[cross]))
P_J, P_test = (sim.particles[i].orbit(primary=sim.particles[0]).P for i in (1, 2))

# analytic period 
libration_P_analytic = P_J/np.sqrt(27*mu/4)

# compute the percent error 
p_err_libration_P = np.abs(libration_P_numeric - libration_P_analytic)/libration_P_analytic*100

print(f"measured synodic period {libration_P_numeric:.1f} yr; "f"predicted {libration_P_analytic:.1f} yr")
print(f'Libration period percent error: {p_err_libration_P:.3f}%')

# plot phi(t) 
# needed to do this on a separate graph because this particle escapes (it is not stable)
fig_l2, ax_l2 = plt.subplots(figsize=(8, 4))
ax_l2.plot(times, phi, lw=0.9, label=r'$\mu = 0.05$')


# prettify plot 
ax_l.set(xlabel="time (yr)", ylabel=r"$\phi_{Particle} - \phi_{Jupiter}$ (deg)", ylim=(50, 70))
ax_l.legend(frameon=False)
ax_l2.set(xlabel="time (yr)", ylabel=r"$\phi_{Particle} - \phi_{Jupiter}$ (deg)", ylim=(-180, 180))
ax_l2.legend(frameon=False)
