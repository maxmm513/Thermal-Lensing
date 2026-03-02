import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ThermalLensLibrary as TL

plt.close('all')

#%% Parameters and Optical System
w0 = 1e-3
zR_num = TL.z_R(w0)       
z0 = -1 * zR_num    
P_list = [249]

m01 = 4e-9
m02 = m01
m03 = m01

eps = 0

f1_dict = 500e-3
f2_dict = -125e-3
dist = f1_dict+f2_dict + eps

f3_dict = 350e-3
dist2 = dist + 3

optics = [
    
    {'z': 0, 'f_base': f1_dict, 'm0': m01, 'name': f'{f1_dict*1e3} mm'},
    {'z': dist, 'f_base': f2_dict, 'm0': m02, 'name': f'{f2_dict*1e3} mm'},
    # {'z': 580e-3, 'f_base': None, 'm0':-4.358e-8, 'name':'AOM'}
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
P_dense = np.linspace(0.2, 1500, 3000)

z0_list = [-3*zR_num, -1*zR_num, 0, 1*zR_num, 3*zR_num]

results = TL.Plot_2Lens_Diagnostics_V2(optics, P_dense, w0, z0_list)

# results['w_out'][i,j]
#   i: z0/zR value
#   j: P value

P0_value, P0_idx, P1_value, P1_idx, wL2_min, wL2_atP1 = TL.Power_minW_L2(results, z0_list)

TL.Galilean_Diagnostic(results, z0_list, dist, P0_idx, P1_idx, z0_idx=1, w0=w0, eps=6.61e-1)

    
#%% sum F1, F2 and idnetify power where afocal condition (C=0) is satisfied

plt.figure()

k=1
EffFocalLength_sum = results['F1'][k,:] + results['F2'][k,:]

mask = np.abs(EffFocalLength_sum) < 1
Pfilt = P_dense[mask]
Fsumfilt = EffFocalLength_sum[mask]

plt.plot(Pfilt,Fsumfilt)
plt.axhline(y=dist, ls='--', color='r')

plt.figure()
plt.plot(P_dense, results['z_out'][k,:])
# plt.axvline(x=1701)

#%%

def FindValsNearZero(arr, eps=5e-3):

    idx = np.where(np.abs(arr) < eps)[0]
    vals = arr[idx]
    
    return idx, vals
      
