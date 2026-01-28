import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ThermalLensLibrary as TL

plt.close('all')

#%% Parameters and Optical System
wavelength = 1064e-9     
w0 = 1e-3               
# P_list = [1,30,75,125,200]
P_list = [10, 50, 100, 250, 500]
m0 = 4e-9

f1_dict = 200e-3
f2_dict = 50e-3
dist = f1_dict+f2_dict

optics = [
    # {'z': 0, 'f_base': 0.250, 'm0': m0, 'name': '250 mm'},
    # {'z': 0.375, 'f_base': 0.125, 'm0': m0, 'name': '125 mm'}
    
    {'z': 0, 'f_base': f1_dict, 'm0': m0, 'name': f'{f1_dict*1e3} mm'},
    {'z': dist, 'f_base': f2_dict, 'm0': m0, 'name': f'{f2_dict*1e3} mm'}
]

z_obs = 0.4

#%%
z_points = np.linspace(0, z_obs, 3000) 

# plt.figure(figsize=(9,4))
plt.figure(figsize=(6,3))
for P in P_list:
    w_z,thermal_f = TL.propagate(optics, z_points, w0, wavelength, P)
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
# Plot waist vs beam power in the final plane of z_obs

# P_scan = np.linspace(min(P_list), max(P_list), 100)
# w_final = np.zeros_like(P_scan)
# for i, P in enumerate(P_scan):
#     w_final[i] = TL.propagate(optics, np.array([z_obs]), w0, wavelength, P)[0]

# plt.figure(figsize=(9,4))
# plt.plot(P_scan, w_final*1e6)
# plt.xlabel('Power (W)', fontsize=13)
# plt.ylabel('Beam radius at final plane (µm)', fontsize=13)
# plt.grid(True, alpha=0.5)
# plt.tight_layout()

#%% Beam behavior after final optic

# find waists at each power after final optic
target_name = optics[-1]['name']

zmin_list, w_min_list = TL.find_waist_after(
    optics,
    target_name=target_name,
    w0=w0,
    wavelength=wavelength,
    P_list=P_list,
    z_max=z_obs
)

zmin_list = zmin_list - optics[-1]['z'] # measure zmin relative to optic
 
z_after_dict = {}
w_after_dict = {}
z_max = z_obs

for P in P_list:
    z_after, w_after = TL.beam_after_last_optic(optics, w0, wavelength, P, z_max=z_max)
    z_after_dict[P] = z_after
    w_after_dict[P] = w_after


# plot beam behavior after final optic
plt.figure(figsize=(6,3))

for j in range(len(P_list)):
    plt.plot((z_after_dict[P_list[j]] - optics[-1]['z'])*1e3, # plot z relative to final optic
             w_after_dict[P_list[j]]*1e6,
             label=f'P={P_list[j]} W, z0={zmin_list[j]*1e3:.1f} mm')
    
    plt.axvline(x=zmin_list[j]*1e3, ymax=0.7, color='C'+str(j), linestyle='--', alpha=0.3)

plt.xlabel('Distance after final optic (mm)')
plt.ylabel('Beam radius (µm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

#%% Find optimal L2 position to collimate at a target power

# P_target = P_list[-1]

# best_z, best_divergence, result = TL.find_best_lens_position_opt(
#     optics,
#     lens_name=optics[-1]['name'],
#     w0=w0,
#     wavelength=wavelength,
#     P=P_target,
#     z_bounds=(0.20, 0.70)
# )

# print(f"Best lens position for P={P_target} W:")
# print(f"   z = {best_z*1e3:.2f} mm")
# print(f"   |dw/dz| = {best_divergence:.3e}")

# z_plot = np.linspace(0.20, 0.70, 200)
# score_plot = [TL.divergence_score(z, optics, optics[-1]['name'], w0, wavelength, P_target)
#               for z in z_plot]

# plt.figure(figsize=(7,4))
# plt.plot(z_plot*1e3, score_plot)
# plt.axvline(best_z*1e3, color='r', linestyle='--')
# plt.xlabel('Lens position (mm)')
# plt.ylabel('|dw/dz| divergence')
# plt.title('Divergence vs lens position')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()

#%% Analytical calculation -- assume collimated input

Pow = np.linspace(1,np.max(P_list)*10,10000)

f1 = optics[0]['f_base']
f2 = optics[1]['f_base']

wL1 = w0
f1_eff = TL.effective_focalLength(f1, Pow, m0, wL1)

wL2 = TL.waist_L2(w0, f1_eff, f1+f2)
f2_eff = TL.effective_focalLength(f2, Pow, m0, wL2)

z0_after, w0_after = TL.waistAndLoc_afterTele(w0, f1_eff, f2_eff, f1+f2)

# inflection power
Pidx = np.argmax(z0_after)
P_inflec = Pow[Pidx]

fig, ax = plt.subplots(1,2, figsize=(6,3))
ax[0].plot(Pow, z0_after*1e3)
# ax[0].axvline(P_inflec, ls='--', alpha=0.3)
ax[0].set_ylabel('Focus after lens (mm)')
ax[0].set_xlabel('Power (W)'); ax[0].grid(True, alpha=0.3)
# ax[0].text(0.45, 0.95,                      
#     f'Inflection P = {P_inflec:.2f} W',
#     transform=ax[0].transAxes,
#     verticalalignment='top',
#     bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)
# )

ax[1].plot(Pow, w0_after*1e6)
ax[1].set_ylabel('Waist after telescope (um)')
ax[1].set_xlabel('Power (W)'); ax[1].grid(True, alpha=0.3)
plt.tight_layout()


print(f'inflection power = {P_inflec:.2f} W')
print(f'Percent change in z0 from maximum: {(np.max(z0_after) - z0_after[-1])/np.max(z0_after)}')
print(f'Percent change in w0: {np.abs(w0_after[-1]-w0_after[0])/w0_after[0]}')

# asymptotic behavior -- need to fix the function
# z0_largeP, w0_largeP = TL.waistAndLoc_asymptotic(w0, f1, f2, f1+f2, m0, wL2, Pow)



#%% Animation

z_plot = np.linspace(0, z_obs, 3000)
z_lens_positions = np.linspace(0.1, 0.4, 120)

# ani = TL.AnimateBeamAfterLastOptic(
#     optics,
#     z_plot,
#     z_lens_positions,
#     lens_to_move=optics[-1]['name'],
#     P_anim=40,
#     w0=w0
# )

P_values = np.linspace(0, 10, 300)

ani = TL.AnimateBeamVsPower(
    optics,
    z_plot,
    P_values,
    w0
)

