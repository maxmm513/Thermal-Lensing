import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ThermalLensLibrary as TL

plt.close('all')

#%% Parameters and Optical System
# w0 = 1.95e-3
w0 = 2.2e-3
w0_err = 0.06e-3

zR_num = TL.z_R(w0)       
z0 = 0 * zR_num    
P_list = [4.5, 37, 81, 124]

m01 = 3.762e-9
m01_err = 0.42e-9

m02 = 2.715e-9
m02_err = 0.718e-9

f1_dict = 304e-3
f2_dict = 122e-3

# dist12 = f1_dict + f2_dict #1.5
dist12 = 0.414

z_obs = dist12 + 0.6
z_points = np.linspace(0, z_obs, 3000) 


optics = [
    {
        'z': 0, 
        'f_base': f1_dict, 
        'm0': m01, 
        'm0_err': m01_err, 
        'name': f'{f1_dict*1e3} mm'
    },
    {
        'z': dist12, 
        'f_base': f2_dict, 
        'm0': m02, 
        'm0_err': m02_err,
        'name': f'{f2_dict*1e3} mm'
    },
]

plt.figure(figsize=(7,4))

for P in P_list:
    # path with  average w0
    w_nom, _ = TL.propagate(optics, z_points, w0, P, z0=z0)
    
    # Upper bound (worst case)
    # use (w0 - error) and (m0 + error) 
    optics_high = [{**o, 'm0': o['m0'] + o['m0_err']} for o in optics]
    w_upper, _ = TL.propagate(optics_high, z_points, w0 - w0_err, P, z0=z0)
    
    # Lower bound (best case)
    # use (w0 + error) and (m0 - error)
    optics_low = [{**o, 'm0': o['m0'] - o['m0_err']} for o in optics]
    w_lower, _ = TL.propagate(optics_low, z_points, w0 + w0_err, P, z0=z0)
    
    line, = plt.plot(z_points*1e3, w_nom*1e3, label=f'P={P}W')

    # Shaded uncertainty region
    plt.fill_between(
        z_points*1e3, 
        w_lower*1e3, 
        w_upper*1e3, 
        color=line.get_color(), 
        alpha=0.15, 
        edgecolor='none'
    )

plt.axvline(optics[-1]['z']*1e3, ls='--', alpha=0.3, color='k')
plt.xlabel('z (mm)')
plt.ylabel('Beam Radius (mm)')
plt.grid(True, alpha=0.3)
plt.legend()

#% load experiment data and compare
statsDF = pd.read_csv('stats_collimationRetry_03122026.csv')
statsDF['Distance'] = statsDF['Distance'] + dist12

j=0
for val, group in statsDF.groupby('Power'):
    plt.errorbar(group['Distance']*1e3, group['Ywidth_mean']*1e3, yerr=group['Ywidth_std']*1e3,
                 xerr = [2]*len(group),
                 marker='o', capsize=3, color='C'+str(j))
    j = j+1

#% Beam behavior after final optic

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
             w_after_dict[P_list[j]]*1e3,
             label=f'P={P_list[j]} W, z0={zmin_list[j]*1e3:.1f} mm')
    
    plt.axvline(x=zmin_list[j]*1e3, ymax=0.7, color='C'+str(j), linestyle='--', alpha=0.3)

plt.xlabel('Distance after final optic (mm)')
plt.ylabel('Beam radius (µm)')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

statsDF['Distance'] = statsDF['Distance'] - dist12

j=0
for val, group in statsDF.groupby('Power'):
    plt.errorbar(group['Distance']*1e3, group['Ywidth_mean']*1e3, yerr=group['Ywidth_std']*1e3,
                 xerr = [2]*len(group),
                 marker='o', capsize=3, color='C'+str(j))
    j = j+1

#%% Monte Carlo

# --- Configuration ---
N_trials = 200         # Number of simulations
P_test = 100           # Power level to simulate (Watts)

all_paths = []

plt.figure(figsize=(10, 5))

# Monte Carlo Loop
for i in range(N_trials):
    
    # sample initial radius from a normal distribution
    w0_sampled = np.random.normal(loc=w0, scale=w0_err)
    
    # sample m0 for each lens individually
    optics_random = []
    for elem in optics:
        m0_sampled = np.random.normal(loc=elem['m0'], scale=elem['m0_err'])
        optics_random.append({**elem, 'm0': m0_sampled})
    
    # Propagate the beam through this specific randomized system
    # NOTE: using the sampled w0 here
    w_sampled, _ = TL.propagate(optics_random, z_points, w0_sampled, P_test, z0=z0)
    all_paths.append(w_sampled)
    
    # Optional: Plot thin lines for the "blur" effect
    # plt.plot(z_points*1e3, w_sampled*1e3, color='royalblue', alpha=0.05, linewidth=0.5)

# Calculate Statistical Bounds
all_paths = np.array(all_paths)
w_mean = np.mean(all_paths, axis=0)
w_95_upper = np.percentile(all_paths, 97.5, axis=0)
w_95_lower = np.percentile(all_paths, 2.5, axis=0)

# 3. Plotting the Comparison
# Plot the 95% Confidence Interval (The Statistical Blur)
plt.fill_between(z_points*1e3, w_95_lower*1e3, w_95_upper*1e3, 
                 color='royalblue', alpha=0.2, label='95% Confidence Interval')

# Plot the Mean Path
plt.plot(z_points*1e3, w_mean*1e3, color='blue', lw=2, label='Mean Simulation')

# --- Overlay Experimental Data ---
# If you have z_exp and w_exp arrays:
# plt.errorbar(z_exp*1e3, w_exp*1e3, yerr=w_meas_err*1e3, fmt='ro', label='Experiment')

# Formatting
for elem in optics:
    plt.axvline(elem['z']*1e3, color='k', linestyle=':', alpha=0.5)

plt.title(f'Monte Carlo at {P_test} W')
plt.xlabel('z (mm)')
plt.ylabel('Beam Radius (mm)')
plt.legend()
plt.grid(True, alpha=0.2)

#%% Monte Carlo multi powers

# --- Configuration ---
w0_nominal = 1.0e-3    
w0_err = 0.06e-3       
P_list = [75]  # Powers to compare
N_trials = 100         
colors = ['tab:blue', 'tab:orange', 'tab:green']

plt.figure(figsize=(12, 6))

for P, color in zip(P_list, colors):
    all_w_P = []
    
    # 1. Run Monte Carlo for this specific Power
    for _ in range(N_trials):
        # Sample w0
        w0_s = np.random.normal(w0_nominal, w0_err)
        
        # Sample m0 for each lens
        optics_s = []
        for o in optics:
            m0_s = np.random.normal(o['m0'], o['m0_err'])
            optics_s.append({**o, 'm0': m0_s})
            
        w_path, _ = TL.propagate(optics_s, z_points, w0_s, P, z0=z0)
        all_w_P.append(w_path)
    
    # 2. Statistics for this Power
    all_w_P = np.array(all_w_P)
    w_mean = np.mean(all_w_P, axis=0)
    w_upper = np.percentile(all_w_P, 97.5, axis=0)
    w_lower = np.percentile(all_w_P, 2.5, axis=0)
    
    # 3. Plotting the "Corridor"
    plt.plot(z_points*1e3, w_mean*1e3, color=color, lw=2, label=f'Mean Path ({P}W)')
    plt.fill_between(z_points*1e3, w_lower*1e3, w_upper*1e3, 
                     color=color, alpha=0.2, edgecolor='none')

# --- 4. Experimental Data Overlay ---
# If you have data for specific powers, plot them here:
# plt.errorbar(z_exp_100, w_exp_100, yerr=w_err, fmt='o', color='tab:orange')

# Optics markers
for elem in optics:
    plt.axvline(elem['z']*1e3, color='black', linestyle='--', alpha=0.3)
    plt.text(elem['z']*1e3, plt.ylim()[1]*0.95, elem['name'], rotation=90, ha='right')

plt.title('Statistical Beam Propagation vs. Laser Power')
plt.xlabel('z (mm)')
plt.ylabel('Beam Radius (mm)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.2)
plt.tight_layout()
