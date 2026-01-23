import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ThermalLensLibrary as TL

plt.close('all')

#%% Parameters and Optical System
wavelength = 1064e-9     
w0 = 1.9e-3               
P_list = [1, 30, 75, 125, 200]
m0 = 4e-9

optics = [
    {'z': 0.150, 'f_base': 0.300, 'm0': m0, 'name': '300 mm'},
    {'z': 0.575, 'f_base': 0.125, 'm0': m0, 'name': '125 mm'}
]

z_obs = 1.5

#%%
    
z_points = np.linspace(0, z_obs, 800) 

plt.figure(figsize=(9,4))
for P in P_list:
    w_z = TL.propagate(optics, z_points, w0, wavelength, P)
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
# Plot final waist vs beam power

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

#%%
target_name = optics[-1]['name']

zmin_list, w_min_list = TL.find_waist_after(
    optics,
    target_name=target_name,
    w0=w0,
    wavelength=wavelength,
    P_list=P_list,
    z_max=1.5
)

# measure zmin relative to optic
zmin_list = zmin_list - optics[-1]['z']

#%%

z_after_dict = {}
w_after_dict = {}

for P in P_list:
    z_after, w_after = TL.beam_after_last_optic(optics, w0, wavelength, P, z_max=1.5)
    z_after_dict[P] = z_after
    w_after_dict[P] = w_after

plt.figure(figsize=(8,4))
for P in P_list:
    plt.plot((z_after_dict[P] - optics[-1]['z'])*1e3,
             w_after_dict[P]*1e6,
             label=f'P={P} W')

plt.xlabel('Distance after final optic (mm)')
plt.ylabel('Beam radius (µm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

#%%

P_target = P_list[-1]

best_z, best_divergence, result = TL.find_best_lens_position_opt(
    optics,
    lens_name=optics[-1]['name'],
    w0=w0,
    wavelength=wavelength,
    P=P_target,
    z_bounds=(0.20, 0.70)
)

print(f"Best lens position for P={P_target} W:")
print(f"   z = {best_z*1e3:.2f} mm")
print(f"   |dw/dz| = {best_divergence:.3e}")

z_plot = np.linspace(0.20, 0.70, 200)
score_plot = [TL.divergence_score(z, optics, optics[-1]['name'], w0, wavelength, P_target)
              for z in z_plot]

plt.figure(figsize=(7,4))
plt.plot(z_plot*1e3, score_plot)
plt.axvline(best_z*1e3, color='r', linestyle='--')
plt.xlabel('Lens position (mm)')
plt.ylabel('|dw/dz| divergence')
plt.title('Divergence vs lens position')
plt.grid(True, alpha=0.3)
plt.tight_layout()


#%% Animate beam radius vs z as the -75 mm lens moves

# lens_to_move = '125 mm'

# # Range of lens positions to animate (meters)
# z_positions_anim = np.linspace(0.377, 0.581, 100)

# # Use a single power setting for animation
# P_anim = P_list[-1]

# TL.AnimateBeamAfterLastOptic(optics, z_positions_anim, lens_to_move, P_anim, w0)