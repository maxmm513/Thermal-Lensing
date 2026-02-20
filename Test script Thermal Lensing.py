import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ThermalLensLibrary as TL

plt.close('all')

#%% Parameters and Optical System
w0 = 1e-3
zR_num = TL.z_R(w0)       
z0 = 0 * zR_num    
P_list = [1, 30, 100, 150, 200]

m01 = 4e-9
m02 = m01
m03 = m01

eps = 0

f1_dict = 300e-3
f2_dict = 125e-3
dist = f1_dict+f2_dict + eps

f3_dict = 350e-3
dist2 = dist + 3

optics = [
    
    {'z': 0, 'f_base': f1_dict, 'm0': m01, 'name': f'{f1_dict*1e3} mm'},
    {'z': dist, 'f_base': f2_dict, 'm0': m02, 'name': f'{f2_dict*1e3} mm'},
    {'z': 580e-3, 'f_base': None, 'm0':-4.358e-8, 'name':'AOM'}
    # {'z': dist+dist2, 'f_base': f3_dict, 'm0': m03, 'name':f'{f3_dict*1e3} mm'}
]

# z_obs = dist2 + 1.5
z_obs = 0.8
z_points = np.linspace(0, z_obs, 3000) 

#%%
# plt.figure(figsize=(9,4))
plt.figure(figsize=(6,3))
for P in P_list:
    w_z,thermal_f = TL.propagate(optics, z_points, w0, P, z0=z0)
    plt.plot(z_points*1e3, w_z*1e6, label=f'P={P} W')

# plot optic locations
for elem in optics:
    plt.axvline(elem['z']*1e3, color='k', linestyle=':', linewidth=0.7)
    plt.text(elem['z']*1e3, plt.ylim()[1]*0.9, elem['name'],
             rotation=90, va='top', ha='center', fontsize=9)

plt.xlabel('z (mm)', fontsize=13)
plt.ylabel('Beam radius (µm)', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


#%%
P_dense = np.linspace(0.2, 1000, 5000)

z0_list = [-3*zR_num, -1*zR_num, 0, 1*zR_num, 3*zR_num]

wint_list = TL.Plot_2Lens_Diagnostics(optics, P_dense, w0, z0_list)