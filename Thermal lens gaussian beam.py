import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

#%% Load Measured data

lensType = 'SPX059AR.1'

if lensType == 'SPX030AR.33':
    material = 'UVFS'
    nominal_f_mm = 350
    m0_lens = 3e-9
    
elif lensType == 'SPX031AR.1':
    material = 'HPFS'
    nominal_f_mm = 500
    m0_lens = 1.033e-9
    
elif lensType == 'SPX059AR.1':
    material = 'HPFS'
    nominal_f_mm = 300
    m0_lens = 2.6e-9
    
elif lensType == 'SPX023AR.1':
    material = 'HPFS'
    nominal_f_mm = 125
    m0_lens = 2.05e-9

folder = 'C:/Users\wmmax\Documents\Lehigh\Sommer Group\Thermal lensing'
filename = 'Focus shift data ' + lensType + ' ' + str(nominal_f_mm) + ' mm lens.csv'

data = pd.read_csv(folder+'/'+filename)
dfA = data[data['Axis'] == 'X']

P_meas = dfA['Power_W']
z0_meas = dfA['z0_mm'] * 1e-3
z0_meas_err = dfA['z0_err_mm'] * 1e-3

#%% Parameters and Optical System
wavelength = 1064e-9     
w0 = 1.9e-3               
P_list = [5, 38, 86, 133]

optics = [
    # {'z': 0.02253, 'f_base': None,  'm0': -5e-10, 'name': 'WP 1'},
    # {'z': 0.05481, 'f_base': None,  'm0':  4.1e-9, 'name': 'Polarizer'},
    # {'z': 0.09784, 'f_base': None,  'm0': -8.63e-9, 'name': 'WP 2'},
    # {'z': 0.122, 'f_base':  z0_meas[0], 'm0': m0_lens, 'name': str(nominal_f_mm)+' mm'},
    # {'z': 0.533, 'f_base': 0.125, 'm0': m0_lens*(3/3.586), 'name': '125 mm'}
    # {'z': 0.378, 'f_base': -0.075, 'm0': m0_lens*(3.2/3.5), 'name': '-75 mm'}
    # {'z':0.35, 'f_base':z0_meas[0], 'm0': m0_lens, 'name': str(nominal_f_mm)+' mm'}
    {'z': 0.122, 'f_base': 0.300, 'm0': 2.60e-9, 'name': '300 mm'},
    {'z': 0.547, 'f_base': 0.125, 'm0': 2.05e-9, 'name': '125 mm'}
]

z_obs = 1.5

#%% Functions
def z_R(w0, wavelength):
    return np.pi * w0**2 / wavelength

def q_at_waist(w0, wavelength):
    return 1j * z_R(w0, wavelength)

def waist_from_q(q, wavelength):
    return np.sqrt(-wavelength / (np.pi * np.imag(1/q)))

def M_free(L):
    return np.array([[1, L],
                     [0, 1]])

def M_lens(f):
    return np.array([[1, 0],
                     [-1 / f, 1]])

def apply_matrix(q, M):
    A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
    return (A*q + B) / (C*q + D)


def propagate(optics, z_points, w0, wavelength, P):
    q = q_at_waist(w0, wavelength)
    z_current = 0
    w_z = np.zeros_like(z_points)

    # Sort optics by z just in case
    optics_sorted = sorted(optics, key=lambda o: o['z'])
    thermal_f = []

    for i, z in enumerate(z_points):
        
        # process all optics up to this z value
        # removing an element from the optics list each iter
        while len(optics_sorted) > 0 and optics_sorted[0]['z'] <= z:
            elem = optics_sorted.pop(0)
            z_elem = elem['z']
            L = z_elem - z_current
            
            # propagate distance between optics
            if L > 0:
                q = apply_matrix(q, M_free(L))
            
            z_current = z_elem

            # compute current beam waist at this optic
            w_here = waist_from_q(q, wavelength)

            # thermal focal length
            # calculate only if m0 is not None
            f_th = np.inf
            if elem['m0'] is not None:
                f_th = (w_here**2) / (elem['m0'] * P)
                thermal_f.append(f_th)

            # combine focal lengths 
            f_effective = None
            if elem['f_base'] is not None and np.isfinite(f_th):
                f_effective = 1 / (1/elem['f_base'] + 1/f_th)
            elif elem['f_base'] is not None:
                f_effective = elem['f_base']
            elif np.isfinite(f_th):
                f_effective = f_th

            # apply lens if finite
            if f_effective is not None and np.isfinite(f_effective):
                q = apply_matrix(q, M_lens(f_effective))
                
        # propagate remaining distance
        L = z - z_current
        q_temp = apply_matrix(q, M_free(L))
        w_z[i] = waist_from_q(q_temp, wavelength)
        
    return w_z

#%%
    
z_points = np.linspace(0, z_obs, 800) 

plt.figure(figsize=(9,4))
for P in P_list:
    w_z = propagate(optics, z_points, w0, wavelength, P)
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
#     w_final[i] = propagate(optics, np.array([z_obs]), w0, wavelength, P)[0]

# plt.figure(figsize=(9,4))
# plt.plot(P_scan, w_final*1e6)
# plt.xlabel('Power (W)', fontsize=13)
# plt.ylabel('Beam radius at final plane (µm)', fontsize=13)
# plt.grid(True, alpha=0.5)
# plt.tight_layout()

#%%
def find_waist_after(optics, target_name, w0, wavelength, P_list, 
                     z_max=2.0, N=4000):

    # Locate the optical element by name
    matches = [o for o in optics if o['name'] == target_name]
    if len(matches) == 0:
        raise ValueError(f"No optic found with name {target_name!r}")
    z_target = matches[0]['z']

    z_points = np.linspace(z_target, z_max, N)

    z_min_list = []
    w_min_list = []

    for P in P_list:
        w_z = propagate(optics, z_points, w0, wavelength, P)

        idx = np.argmin(w_z)
        z_min_list.append(z_points[idx])
        w_min_list.append(w_z[idx])

    return np.array(z_min_list), np.array(w_min_list)

target_name = optics[-1]['name']

zmin_list, w_min_list = find_waist_after(
    optics,
    target_name=target_name,
    w0=w0,
    wavelength=wavelength,
    P_list=P_list,
    z_max=1.5
)

# measure zmin relative to optic
zmin_list = zmin_list - optics[-1]['z']

#%% Scan different m0 values for a lens
# m0_scan = np.linspace(m0_lens*0.8, m0_lens*1.2, 100)
m0_scan = [0.8e-9, 0.9e-9, 1e-9, 1.25e-9, 1.5e-9, 2e-9]

results = {}

for m0_try in m0_scan:
    # copy of optics
    optics_test = []
    for elem in optics:
        optics_test.append(elem.copy())

    # update m0 for the appropriate element
    for elem in optics_test:
        if elem["name"] == target_name:
            elem["m0"] = m0_try

    # compute zmin for this m0 value
    zmin_list, _ = find_waist_after(
        optics_test,
        target_name=target_name,
        w0=w0,
        wavelength=wavelength,
        P_list=P_list,
        z_max=1
    )

    # convert to relative distance after the optic
    zmin_list = zmin_list - [o for o in optics_test if o["name"] == target_name][0]["z"]

    results[m0_try] = zmin_list


plt.figure(figsize=(6,4))

plt.errorbar(P_meas, z0_meas, yerr=z0_meas_err, color='k', fmt='o', capsize=3, label='measured')

for m0_try, zmin_list in results.items():
    plt.plot(P_list, zmin_list,
             label=f"m0 = {m0_try:.2e}")

plt.xlabel('Power (W)')
plt.ylabel('Waist position')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()


# Compute SSE for each m0
sse = {}
for m0_try, zmin_list in results.items():
    sse[m0_try] = np.sum( (zmin_list - z0_meas)**2 )

best_m0 = min(sse, key=sse.get)
print("Best m0 =", best_m0)

#%%
def beam_after_last_optic(optics, w0, wavelength, P, z_max=1.5, N=2000):
 
    optics_sorted = sorted(optics, key=lambda o: o['z'])
    z_last = optics_sorted[-1]['z']

    # z vals AFTER the final optic
    z_points = np.linspace(z_last, z_max, N)
    w_after = propagate(optics, z_points, w0, wavelength, P)

    return z_points, w_after

z_after_dict = {}
w_after_dict = {}

for P in P_list:
    z_after, w_after = beam_after_last_optic(optics, w0, wavelength, P, z_max=1.5)
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
from scipy.optimize import minimize_scalar

def divergence_score(z_lens, optics, lens_name, w0, wavelength, P):

    optics_test = [o.copy() for o in optics]

    # Move the specific lens
    for o in optics_test:
        if o['name'] == lens_name:
            o['z'] = z_lens

    # Propagate after final optic
    z_after, w_after = beam_after_last_optic(
        optics_test, w0, wavelength, P, z_max=0.75
    )

    # Fit w(z) to a straight line, divergence is the slope
    slope = np.polyfit(z_after, w_after, 1)[0]
    return abs(slope)

def find_best_lens_position_opt(
    optics, lens_name, w0, wavelength, P,
    z_bounds=(0, 1)  # meters
):

    res = minimize_scalar(
        lambda z: divergence_score(z, optics, lens_name, w0, wavelength, P),
        bounds=z_bounds,
        method='bounded'
    )

    return res.x, res.fun, res

P_target = P_list[-1]

best_z, best_divergence, result = find_best_lens_position_opt(
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
score_plot = [divergence_score(z, optics, optics[-1]['name'], w0, wavelength, P_target)
              for z in z_plot]

plt.figure(figsize=(7,4))
plt.plot(z_plot*1e3, score_plot)
plt.axvline(best_z*1e3, color='r', linestyle='--')
plt.xlabel('Lens position (mm)')
plt.ylabel('|dw/dz| divergence')
plt.title('Divergence vs lens position (verification)')
plt.grid(True, alpha=0.3)
plt.tight_layout()


#%% Animate beam radius vs z as the -75 mm lens moves

# import matplotlib.animation as animation

# lens_to_move = '-75 mm'

# # Range of lens positions to animate (meters)
# z_positions_anim = np.linspace(0.377, 0.381, 150)

# # Use a single power setting for animation (or choose one)
# P_anim = P_list[-1]   # highest power, for example

# fig, ax = plt.subplots(figsize=(9,5))

# line, = ax.plot([], [], lw=2)
# lens_marker = ax.axvline(0, color='r', linestyle='--', alpha=0.6)

# ax.set_xlim(z_points.min()*1e3, z_points.max()*1e3)
# ax.set_ylim(0, 3000)

# ax.set_xlabel('z (mm)')
# ax.set_ylabel('Beam radius (µm)')
# ax.set_title('Beam radius vs z as -75 mm lens position changes')

# text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=12)

# def init():
#     line.set_data([], [])
#     return line, lens_marker, text

# def update(frame):
#     z_new = z_positions_anim[frame]

#     # Copy optics and move lens
#     optics_test = [o.copy() for o in optics]
#     for o in optics_test:
#         if o['name'] == lens_to_move:
#             o['z'] = z_new

#     # Propagate
#     w_z = propagate(optics_test, z_points, w0, wavelength, P_anim)

#     # Update plot
#     line.set_data(z_points*1e3, w_z*1e6)
#     lens_marker.set_xdata([z_new*1e3, z_new*1e3])
#     text.set_text(f"{lens_to_move} at z = {z_new*1e3:.2f} mm")

#     return line, lens_marker, text

# ani = animation.FuncAnimation(
#     fig, update, frames=len(z_positions_anim), init_func=init,
#     interval=200, blit=True
# )

# plt.tight_layout()

# ani.save("beam_radius_animation.gif", fps=20)
