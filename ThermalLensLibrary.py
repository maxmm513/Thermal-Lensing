import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import matplotlib.animation as animation



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

def beam_after_last_optic(optics, w0, wavelength, P, z_max=1.5, N=2000):
 
    optics_sorted = sorted(optics, key=lambda o: o['z'])
    z_last = optics_sorted[-1]['z']

    # z vals AFTER the final optic
    z_points = np.linspace(z_last, z_max, N)
    w_after = propagate(optics, z_points, w0, wavelength, P)

    return z_points, w_after

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



#%% Animation

def AnimateBeamAfterLastOptic(optics, z_positions, lens_to_move, P_anim, w0, wavelength=1064e-9):
    
    fig, ax = plt.subplots(figsize=(9,5))

    line, = ax.plot([], [], lw=2)
    lens_marker = ax.axvline(0, color='r', linestyle='--', alpha=0.6)

    ax.set_xlim(z_positions.min()*1e3, z_positions.max()*1e3)
    ax.set_ylim(0, 3000)

    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Beam radius (µm)')
    ax.set_title(f'Beam radius vs. z as {lens_to_move} lens position changes')

    text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=12)


    def init():
        line.set_data([], [])
        return line, lens_marker, text
    
    def update(frame):
        z_new = z_positions[frame]
    
        # Copy optics and move lens
        optics_test = [o.copy() for o in optics]
        for o in optics_test:
            if o['name'] == lens_to_move:
                o['z'] = z_new
    
        # Propagate
        w_z = propagate(optics_test, z_positions, w0, wavelength, P_anim)
    
        # Update plot
        line.set_data(z_positions*1e3, w_z*1e6)
        lens_marker.set_xdata([z_new*1e3, z_new*1e3])
        text.set_text(f"{lens_to_move} at z = {z_new*1e3:.2f} mm")
    
        return line, lens_marker, text
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(z_positions), init_func=init,
        interval=10, blit=True
    )

    plt.tight_layout()