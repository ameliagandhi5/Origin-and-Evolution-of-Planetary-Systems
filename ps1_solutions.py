#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:49:11 2026

@author: ameliagandhi
"""

# imports 
import numpy as np 
import matplotlib.pyplot as plt 
from astrotools.cloud.collapse import *
from astrotools import constants as c
import pandas as pd 
from scipy.optimize import curve_fit

# define fit 
def power_fit(x, a, b):
    fit = a*x**b
    return fit 

#%%
# 2: reference core 

# define variables
temperature = 10 # Kelvin 
n_h2 = 10**11 # m^-3
mass = 3*c.M_SUN # kg
R_0 = 0.047*c.PC # m 

# convert number density to mass density 
density = number_density_to_mass_density(n_h2)

# calculate the Jeans mass 
M_J = jeans_mass(temperature, density) 
print(f'Jeans Mass (kg): {M_J:.2e}')

# calculate the Jeans length 
# create a function for jeans length 
def jeans_length(temperature, density): 
    L_J = np.sqrt(c.K_B*temperature*np.pi/(c.MU_CLOUD*c.M_H*c.MU_H2*c.G*density)) 
    return L_J 

L_J = jeans_length(temperature, density) # m 
L_J_km = L_J/1e3
print(f'Jeans Length (km): {L_J_km:.2e}')

# calculate the free fall time 
t_ff = free_fall_time(density) 
t_ff_yr = t_ff/(60*60*24*365*1e6) # convert to Myr 
print(f'Free Fall Time (Myr): {t_ff_yr:.2f}')

# calculate MJ/M 
ratio_M = M_J/mass
print(f'M_J/M: {ratio_M:.2f}')

#%%
# 3: characteristic mass 

# import the data 
catalogue = pd.read_csv('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/Electives/Origin Evolution of Planetary Systems/astr5820/Datasets/herschel_core_catalog.csv')

# filter out the prestellar cores 
prestellar_catalogue = catalogue[catalogue['core_type'] == 'prestellar']

# convert everything to proper units to use astrotools 
prestellar_mass_kg = prestellar_catalogue['M_core_msun']*c.M_SUN
prestellar_R_obs_m = prestellar_catalogue['R_deconv_pc']*c.PC
prestellar_nH2_per_m3 = prestellar_catalogue['n_H2_avd_cm3']*(100**3)
prestellar_temperature = prestellar_catalogue['T_dust_K']

# convert number density to mass density 
prestellar_density = number_density_to_mass_density(prestellar_nH2_per_m3)

# part a: compute MJ for every core 
prestellar_M_J = jeans_mass(prestellar_temperature, prestellar_density) 

# plot a histogram of the Jeans Masses 
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
bins = np.arange(-1.5, 1.5, .2)
counts_core, bins_core, patches_core = ax.hist(np.log10(prestellar_catalogue['M_core_msun']), color='blue', alpha=.5, bins = bins, label='Core Mass')
counts_jeans, bins_jeans, patches_jeans = ax.hist(np.log10(prestellar_M_J/c.M_SUN), color='green', alpha=.5,  bins = bins, label='Jeans Mass')

# find the peaks of the histogram and plot 
peak_core = bins_core[np.where(counts_core==np.max(counts_core))]+.1
peak_jeans = bins_jeans[np.where(counts_jeans==np.max(counts_jeans))]+.1

ax.axvline(peak_core, label=r'Peak Core Mass ($log(M_{solar})$):' + f'{peak_core[0]:.2f}', c='blue')
ax.axvline(peak_jeans, label=r'Peak Jeans Mass ($log(M_{solar})$):' + f'{peak_jeans[0]:.2f}', c='green')

# report the ratio of these peaks in the histogram 
ratio_histograms = 10**peak_jeans[0]/10**peak_core[0]

print(f'ratio histograms: {ratio_histograms:.2f}')

# report the median of the core and jeans mass 
median_ratio = np.median((prestellar_M_J/c.M_SUN)/prestellar_catalogue['M_core_msun']) 
print(f'Median M_J/M: {median_ratio:.2f}')

# pretty stuff 
ax.set_title('Distribution of the Jeans Mass for Prestellar\nCores in the Hershel Gould Belt Survey')
ax.set_xlabel(r'Jeans Mass ($log(M_{solar})$)')
ax.set_ylabel('Number of Clouds')
fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/Electives/Origin Evolution of Planetary Systems/HW Figs/hw1_jeansmasshist.png')

#%%

# 4: collapse timescale 
N_II = 622
t_II = 2 # in Myr 

# 4.1 
# make a list of density thresholds 
n_thresholds = np.linspace(1e10, 1e12, 100)

# find the number of cores with density thresholds between 10^10 and 10^12 m^-3 
N_thresholds = np.array([np.sum((prestellar_nH2_per_m3 > dt)) for dt in n_thresholds]) 

# compute t_life 
t_l = N_thresholds/N_II*t_II # in Myr

# 4.2
# find t_ff at each threshold density 
density_thresholds = number_density_to_mass_density(n_thresholds) # find the mass density at each threshold from n 
t_ff_thresholds = free_fall_time(density_thresholds)/(60*60*24*365*1e6) # in Myr 

### ASIDE 
# make a table of n thresholds and t_l 
fig_t, ax_t = plt.subplots(figsize=(5, 10)) 
ax_t.axis('off')

table_data = pd.DataFrame({r'Density Thresholds ($m^{-3}$)': n_thresholds, 
                           'Inferred Lifetime (Myr)': t_l, 
                          'Free Fall Time (Myr)': t_ff_thresholds})

# put the datapoints in the right notation 
table_data[r'Density Thresholds ($m^{-3}$)'] = table_data[r'Density Thresholds ($m^{-3}$)'].map(lambda x: f'{x:.1e}')
table_data['Inferred Lifetime (Myr)'] = table_data['Inferred Lifetime (Myr)'].map(lambda x: f'{x:.2f}')
table_data['Free Fall Time (Myr)'] = table_data['Free Fall Time (Myr)'].map(lambda x: f'{x:.2f}')

# write the table 
table = ax_t.table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='center')


# 4.3
# plot tff and tl 
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
fig.subplots_adjust(left=.15, right=.99, bottom=0.15, top=.99)
ax.scatter(t_ff_thresholds, t_l, s=5, c='k')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Free Fall Time (Myr)')
ax.set_ylabel('Inferred Lifetime (Myr)')

# 4.4 
# perform power law fit 
popt, pcov = curve_fit(power_fit, t_ff_thresholds, t_l, p0=[1,1])

a = popt[0]; b = popt[1]

fit = a * t_ff_thresholds**b

ax.plot(t_ff_thresholds, t_ff_thresholds*t_l[0], c='blue', linestyle='--')
ax.text(0.5, 0.28, r'log($t_{life}$)' + f'= log({a:.2f})' + r'log($t_{ff}$)', c='blue', transform=ax.transAxes)

ax.plot(t_ff_thresholds, fit, c='green', linestyle='--')
ax.text(0.28, 0.92, r'log($t_{life}$)' + f'= log({a:.2f}) + {b:.2f}'+r'log($t_{ff}$)', c='green', transform=ax.transAxes)
ax.tick_params(axis='x', which='both', labelrotation=20)

# calculate the mean ratio t_life/t_ff
mean_ratio_t = np.mean(t_l/t_ff_thresholds) 
print(f'Mean Ratio t_life/t_ff: {mean_ratio_t:.2f}')
plt.savefig('/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/Electives/Origin Evolution of Planetary Systems/HW Figs/hw1_tf_tl.png')
