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

z0_list = [-1*zR_num, 0, 1*zR_num]

results = TL.Plot_2Lens_Diagnostics_V2(optics, P_dense, w0, z0_list)

# results['w_out'][i,j]
#   i: z0/zR value
#   j: P value

wL2_min = []
P0_value = []
P0_idx = []

for i in range(len(z0_list)):
    
    # minimum w_L2
    wL2_minVal = min(results['w_L2'][i,:])
    wL2_minVal_Pidx = np.argmin(results['w_L2'][i,:])
    wL2_minVal_P = results['P_values'][wL2_minVal_Pidx]
    
    wL2_min.append(wL2_minVal)
    P0_value.append(wL2_minVal_P)
    P0_idx.append(wL2_minVal_Pidx)
    
wL2_min = np.array(wL2_min); P0_value = np.array(P0_value); P0_idx = np.array(P0_idx)

zR_L2 = np.pi * wL2_min**2 / 1064e-9

P1_value = []
P1_idx = []
wL2_atP1 = []

for i in range(len(z0_list)):
    
    theta_minVal = np.min(results['theta'][i,:])    
    theta_minVal_Pidx = np.argmin(results['theta'][i,:])
    theta_minVal_P = results['P_values'][theta_minVal_Pidx]    
    
    P1_value.append(theta_minVal_P)
    P1_idx.append(theta_minVal_Pidx)
    
    wL2_atP1_val = results['w_L2'][i, theta_minVal_Pidx]
    wL2_atP1.append(wL2_atP1_val)



#%%

plt.figure()

for j in range(len([0,1,2])):
    
    plt.plot(results['P_values'], results['w_L2'][j,:]*1e6, color='C'+str(j))
    plt.grid(True, alpha=0.3)
    
    # plt.axvline(x=P0_value[j], color='C'+str(j), ls='--', alpha=0.3)
    plt.axvline(x=P1_value[j], color='C'+str(j), ls='--', alpha=0.3)
    plt.axhline(y=wL2_atP1[j]*1e6, color='C'+str(j), ls='--', alpha=0.3)
    
#%%

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

k=0
F1 = results['F1'][k,:]
F2 = results['F2'][k,:]

C = TL.C_TelescopeRTM(F1, F2, dist)
denom = TL.q_out_denom(w0, F1, F2, dist, z0=z0)

fig, ax = plt.subplots(1,3, figsize=(12,4))
ax[0].plot(results['P_values'], C)
ax[0].axhline(y=0, ls='--', alpha=0.5, color='k')
ax[0].set_ylabel('C')
ax[0].grid(True,alpha=0.3)

ax[1].plot(results['P_values'], denom)
ax[1].set_ylabel('$q_{out}$ denominator')
ax[1].grid(True,alpha=0.3)

ax[2].plot(results['P_values'], results['z_out'][k,:])
ax[2].set_ylabel('$z_0^\prime$')
ax[2].grid(True,alpha=0.3)
plt.tight_layout()

Cmin_idx, Cmax_idx,_,_ = TL.RelativeExtrema(C)
Qmin_idx, Qmax_idx,_,_ = TL.RelativeExtrema(denom)

Cnear0, _ = FindValsNearZero(C)
# Qnear0, _ = FindValsNearZero(denom, eps=np.min(denom)*1.001)


for subplot in ax:
    PPP = results['P_values']

    for j in range(len(Cnear0)):
        subplot.axvline(x=PPP[Cnear0[j]], ls='--', alpha=0.3, label='P when C=0')

    for m in range(len(Qmin_idx)):
        subplot.axvline(x=PPP[Qmin_idx[m]], ls='--', alpha=0.3, label='P when $q_{out}$ denom $\\approx$ 0', color='C1')
    
    subplot.legend()        
    
#%%
