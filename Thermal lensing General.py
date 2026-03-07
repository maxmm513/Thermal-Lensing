import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ThermalLensLibrary as TL

plt.close('all')

#%% Parameters and Optical System
w0 = 1e-3
zR_num = TL.z_R(w0)       
z0 = 0 * zR_num    
P_list = [0.1, 1, 25, 50, 100, 200]

m01 = 4e-9
m02 = m01
m03 = m01

f1_dict = 500e-3
f2_dict = -125e-3
f3_dict = 350e-3

dist12 = 1.5
dist23 = 1.5

optics = [
    # {'z': 0, 'f_base': 0.250, 'm0': m0, 'name': '250 mm'},
    # {'z': 0.375, 'f_base': 0.125, 'm0': m0, 'name': '125 mm'}
    
    {'z': 0, 'f_base': f1_dict, 'm0': m01, 'name': f'{f1_dict*1e3} mm'},
    {'z': dist12, 'f_base': f2_dict, 'm0': m02, 'name': f'{f2_dict*1e3} mm'},
    {'z': dist12+dist23, 'f_base': f3_dict, 'm0': m03, 'name':f'{f3_dict*1e3} mm'}
]

z_obs = dist12 + dist23 + 1
z_points = np.linspace(0, z_obs, 8000) 

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
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()


#%% Beam behavior after final optic

# find waists at each power after final optic
target_name = optics[-1]['name']

zmin_list, w_min_list = TL.find_waist_after(
    optics,
    target_name=target_name,
    w0=w0,
    P_list=P_list,
    z_max=z_obs,
    z0=z0
)

zmin_list = zmin_list - optics[-1]['z'] # measure zmin relative to optic
 
z_after_dict = {}
w_after_dict = {}
z_max = z_obs

for P in P_list:
    z_after, w_after = TL.beam_after_last_optic(optics, w0, P, z0=z0, z_max=z_max)
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
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

#%%
P_dense = np.linspace(0.2, 200, 1000)

z0_list = [-3*zR_num, -1*zR_num, 0, 1*zR_num, 3*zR_num]
TL.Plot_2Lens_Diagnostics(optics, P_dense, w0, z0_list)

D_list = [0.25, 0.75, 1, 2.5]
TL.Plot_3Lens_DistScan(optics, P_dense, w0, z0, D_list, delta_focus=0)

#%% Run the Grid Search and Plot

P_rms = np.linspace(0.1,200,25)

# parameter space to explore (m)
grid_points = 250
d12_vals = np.linspace(0.1, 1.5, grid_points)
d23_vals = np.linspace(0.1, 1.5, grid_points)

f_nominal = (f1_dict, f2_dict, f3_dict)
m0_vals = (m01, m02, m03)

# RMS of z0'' and w0'' separately
# D12, D23, z0_rms, w0_rms = TL.calculate_rms_drift_2d(
#     d12_vals, d23_vals, f_nominal, m0_vals, P_rms, w0, z0_in=z0
# )

# combined RMS
D12, D23, z0_rms, w0_rms, combined_err = TL.optimize_thermal_stability(
    d12_vals, d23_vals, f_nominal, m0_vals, P_list, w0, 
    z0_in=0,
    weight_z=0.5,
    weight_w=0.5
)

TL.Plot_CombinedScore(D12, D23, combined_err)

#%%
best_configs, worst_configs = TL.get_extreme_rms_combinations(
    D12, D23, combined_err, num_points=5, min_spacing=0.2)

print("--- TOP 5 MOST STABLE CONFIGURATIONS ---")
for i, config in enumerate(best_configs):
    print(f"{i+1}. d12 = {config['d12']:.3f} m, d23 = {config['d23']:.3f} m | RMS = {config['rms']*1e3:.2f} mm")

print("\n--- TOP 5 LEAST STABLE CONFIGURATIONS ---")
for i, config in enumerate(worst_configs):
    print(f"{i+1}. d12 = {config['d12']:.3f} m, d23 = {config['d23']:.3f} m | RMS = {config['rms']*1e3:.2f} mm")



#%% Animation
import matplotlib.animation as animation


z_plot = np.linspace(0, z_obs, 2000)
P_values = np.linspace(1, 500, 100)

# z_lens_positions = np.linspace(0.1, 0.4, 120)

# ani = TL.AnimateBeamAfterLastOptic(
#     optics,
#     z_plot,
#     z_lens_positions,
#     lens_to_move=optics[-1]['name'],
#     P_anim=40,
#     w0=w0
# )

# ani = TL.AnimateBeamVsPower(
#     optics,
#     z_plot,
#     P_values,
#     w0
# )


# f1_eff = TL.effective_focalLength(f1, P_values, m0, wL1)

# wL2 = TL.waist_L2(w0, f1_eff, f1+f2, z0=z0)
# f2_eff = TL.effective_focalLength(f2, P_values, m0, wL2)

# z0_after, w0_after = TL.waistAndLoc_afterTele(w0, f1_eff, f2_eff, f1+f2)


# ani = TL.AnimateBeamAndFocusVsPower(
#     optics,
#     z_plot,
#     P_values,
#     w0,
#     z0=z0_ani
# )


z0_list = [-3*zR_num, -1*zR_num, 0*zR_num, 1*zR_num, 3*zR_num]
# ani = TL.AnimateBeamVsPowerMultipleZ0(
#         optics,
#         z_plot,
#         P_values,
#         w0,
#         z0_list
#     )

L1_dist = 1.2
L2_positions = np.linspace(L1_dist+0.2, L1_dist+0.7, 200)

ani = TL.AnimateBeamMultiPower(
        optics,
        z_plot,
        L2_positions,
        optics[-1]['name'],      
        [0, 20, 50, 100, 150],            # list of powers
        w0,
        z0=0)

# ani.save("beam_power_animation.gif", fps=20)

#%% Find optimal L2 position to collimate at a target power

# P_target = P_list[-1]

# best_z, best_divergence, result = TL.find_best_lens_position_opt(
#     optics,
#     lens_name=optics[-1]['name'],
#     w0=w0,
#     P=P_target,
#     z0=z0,
#     z_bounds=(0.20, 0.70)
# )

# print(f"Best lens position for P={P_target} W:")
# print(f"   z = {best_z*1e3:.2f} mm")
# print(f"   |dw/dz| = {best_divergence:.3e}")

# z_plot = np.linspace(0.20, 0.70, 200)
# score_plot = [TL.divergence_score(z, optics, optics[-1]['name'], w0, P_target, z0=z0)
#               for z in z_plot]

# plt.figure(figsize=(7,4))
# plt.plot(z_plot*1e3, score_plot)
# plt.axvline(best_z*1e3, color='r', linestyle='--')
# plt.xlabel('Lens position (mm)')
# plt.ylabel('|dw/dz| divergence')
# plt.title('Divergence vs lens position')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()