import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import matplotlib.animation as animation

wavelength = 1064e-9

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
                     [-1/f, 1]])

def apply_matrix(q, M):
    A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
    return (A*q + B) / (C*q + D)

def effective_focalLength(f,P,m0,beamRadius):
    alpha = m0*f / beamRadius**2
    f_eff = f / (1+alpha*P)
    return f_eff

# Returns the complex q parameter immediately AFTER the last optic.
def q_after_last_optic(optics, w0, wavelength, P, z0=0):
    zR = z_R(w0, wavelength)
    q = z0 + 1j*zR
    z_current = 0

    optics_sorted = sorted([o.copy() for o in optics], key=lambda o: o['z'])

    for elem in optics_sorted:
        z_elem = elem['z']

        # free space to optic
        L = z_elem - z_current
        if L > 0:
            q = apply_matrix(q, M_free(L))

        z_current = z_elem

        # beam size at optic
        w_here = waist_from_q(q, wavelength)

        # thermal focal length
        f_th = np.inf
        if elem['m0'] is not None:
            f_th = (w_here**2) / (elem['m0'] * P)

        # effective lens
        if elem['f_base'] is not None and np.isfinite(f_th):
            f_eff = 1 / (1/elem['f_base'] + 1/f_th)
        elif elem['f_base'] is not None:
            f_eff = elem['f_base']
        elif np.isfinite(f_th):
            f_eff = f_th
        else:
            continue

        q = apply_matrix(q, M_lens(f_eff))

    return q

#%% use Gaussian beam analysis to simulate general optical system

def propagate(optics, z_points, w0, wavelength, P, z0=0):
    """
    General Gaussian propagation with waist located z0 BEFORE first optic.
    z0 > 0  => converging beam at first optic
    z0 < 0  => diverging beam at first optic
    """

    zR = z_R(w0, wavelength)

    # ---- THIS IS THE KEY FIX ----
    # q at first optic, not at waist
    q = z0 + 1j*zR
    z_current = 0

    w_z = np.zeros_like(z_points)

    optics_sorted = sorted(optics, key=lambda o: o['z'])
    thermal_f = []

    for i, z in enumerate(z_points):

        while len(optics_sorted) > 0 and optics_sorted[0]['z'] <= z:
            elem = optics_sorted.pop(0)
            z_elem = elem['z']
            L = z_elem - z_current

            if L > 0:
                q = apply_matrix(q, M_free(L))

            z_current = z_elem

            w_here = waist_from_q(q, wavelength)

            f_th = np.inf
            if elem['m0'] is not None:
                f_th = (w_here**2) / (elem['m0'] * P)
                thermal_f.append(f_th)

            if elem['f_base'] is not None and np.isfinite(f_th):
                f_effective = 1 / (1/elem['f_base'] + 1/f_th)
            elif elem['f_base'] is not None:
                f_effective = elem['f_base']
            elif np.isfinite(f_th):
                f_effective = f_th
            else:
                f_effective = None

            if f_effective is not None and np.isfinite(f_effective):
                q = apply_matrix(q, M_lens(f_effective))

        L = z - z_current
        q_temp = apply_matrix(q, M_free(L))
        w_z[i] = waist_from_q(q_temp, wavelength)

    return w_z, thermal_f


def find_waist_after(optics, target_name, w0, wavelength, P_list, z0=0,
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
        w_z, _ = propagate(optics, z_points, w0, wavelength, P, z0=z0)
        idx = np.argmin(w_z)
        z_min_list.append(z_points[idx])
        w_min_list.append(w_z[idx])

    return np.array(z_min_list), np.array(w_min_list)

def beam_after_last_optic(optics, w0, wavelength, P, z0=0, z_max=1.5, N=2000):
 
    optics_sorted = sorted(optics, key=lambda o: o['z'])
    z_last = optics_sorted[-1]['z']

    # z vals AFTER the final optic
    z_points = np.linspace(z_last, z_max, N)
    w_after, _ = propagate(optics, z_points, w0, wavelength, P, z0=z0)

    return z_points, w_after

def divergence_score(z_lens, optics, lens_name, w0, wavelength, P, z0=0):

    optics_test = [o.copy() for o in optics]

    # Move the specific lens
    for o in optics_test:
        if o['name'] == lens_name:
            o['z'] = z_lens

    # Propagate after final optic
    z_after, w_after = beam_after_last_optic(
        optics_test, w0, wavelength, P, z0=z0, z_max=0.75
    )

    # Fit w(z) to a straight line, divergence is the slope
    slope = np.polyfit(z_after, w_after, 1)[0]
    return abs(slope)

def find_best_lens_position_opt(
    optics, lens_name, w0, wavelength, P, z0=0,
    z_bounds=(0, 1)  # meters
):

    res = minimize_scalar(
        lambda z: divergence_score(z, optics, lens_name, w0, wavelength, P, z0=z0),
        bounds=z_bounds,
        method='bounded'
    )

    return res.x, res.fun, res

#%% Analytical Gaussian beam calculation

# Telescope lenses separated by distance d

def waist_L1(w0, z0=0):
    zR = z_R(w0, wavelength)
    wL1 = w0/zR * np.sqrt(z0**2 + zR**2)
    return wL1

# input effective focal length for L1
def waist_L2(w0, F1, d, z0=0):
    zR = z_R(w0, wavelength)
    delta = (F1-z0)**2 + zR**2
    A = d + F1/delta * (z0*(F1-z0)-zR**2)
    B = F1**2 * zR / delta
    wL2 = w0*np.sqrt(delta*(A**2 + B**2)) / (F1*zR)
    # wL2 = w0/(F1*zR) * np.sqrt( d**2 * F1**2 + zR**2 * (d-F1)**2)
    return wL2

# input effective focal lengths for both lenses
def waistAndLoc_afterTele(w0, F1, F2, d, z0=0):
    zR = z_R(w0, wavelength)
    delta = (F1-z0)**2 + zR**2
    A = d + F1/delta * (z0*(F1-z0)-zR**2)
    B = F1**2 * zR / delta
    gamma = (F2-A)**2 + B**2
    
    z0_prime = F2*(B**2 - A*(F2-A)) / gamma
    w0_prime = w0*np.abs(F1)*np.abs(F2) / np.sqrt(delta*gamma)
    
    # delta = F1**2 + zR**2
    # alpha = d - (F1*zR**2)/delta
    # beta = F1**2 * zR/delta
    # z0 = -F2*(alpha*(F2-alpha) - beta**2) / ((F2-alpha)**2 + beta**2 )
    # w0 = np.sqrt(wavelength/np.pi * F2**2*beta / ((F2-alpha)**2 + beta**2) )
    
    return z0_prime, w0_prime

def effective_focalLength_largeP(f,m0,beamRadius,P):
    
    alpha = m0*f/beamRadius**2
    return f / (alpha*P)

# input nominal focal lengths
def waistAndLoc_asymptotic(w0, f1, f2, d, m0, wL2, P):
    
    zR = z_R(w0, wavelength)
    
    F1_largeP = effective_focalLength_largeP(f1, m0, w0, P)
    F2_largeP = effective_focalLength_largeP(f2, m0, wL2, P)
    
    delta = F1_largeP**2 + zR**2
    A = d - F1_largeP*zR**2 / delta
    B = F1_largeP**2 * zR/delta
    
    z0_largeP = -F2_largeP*(A*(F2_largeP-A) - B**2) / ((F2_largeP-A)**2 + B**2 )
    w0_largeP = np.sqrt(wavelength/np.pi * F2_largeP**2*B / ((F2_largeP-A)**2 + B**2) )

    return z0_largeP, w0_largeP

def FindExtrema(x,y):
    
    dy = np.gradient(y, x)

    sign = np.sign(dy)
    sign_change = np.diff(sign)
    max_idx = np.where(sign_change < 0)[0] + 1
    min_idx = np.where(sign_change > 0)[0] + 1

    # threshold = 1e-10
    # max_idx = [i for i in max_idx if abs(dy[i-1]) > threshold and abs(dy[i+1]) > threshold]
    # min_idx = [i for i in min_idx if abs(dy[i-1]) > threshold and abs(dy[i+1]) > threshold]

    x_min = x[min_idx]
    x_max = x[max_idx]

    y_min = y[min_idx]
    y_max = y[max_idx]
    
    return x_min, x_max, y_min, y_max

def Plot_z0w0_afterTelescope(P, z0_after, w0_after):
        
    fig, ax = plt.subplots(1,2, figsize=(6,3))
    
    ax[0].plot(P, z0_after*1e3)
    ax[0].set_ylabel('Focus after lens (mm)');
    ax[0].set_xlabel('Power (W)');
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(P, w0_after*1e6)
    ax[1].set_ylabel('Waist after telescope (um)'); 
    ax[1].set_xlabel('Power (W)'); 
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()        
    
#%%

def apply_thermal_lens(q_in, wavelength, f_base, m0, P,
                       tol=1e-12, max_iter=30):
   

    # Initial guess: assume no thermal contribution
    if f_base is not None:
        q_guess = apply_matrix(q_in, M_lens(f_base))
    else:
        q_guess = q_in

    for _ in range(max_iter):

        # Beam radius AFTER the lens (depends on lens strength)
        w = waist_from_q(q_guess, wavelength)

        # Thermal focal length from that beam size
        if m0 is not None:
            f_th = w**2 / (m0 * P)
        else:
            f_th = np.inf

        # Combine base and thermal lens
        if f_base is not None and np.isfinite(f_th):
            f_eff = 1 / (1/f_base + 1/f_th)
        elif f_base is not None:
            f_eff = f_base
        else:
            f_eff = f_th

        # Apply that lens to original incoming q
        q_new = apply_matrix(q_in, M_lens(f_eff))

        # Check convergence
        if np.abs(q_new - q_guess) < tol:
            return q_new, f_eff

        q_guess = q_new

    # Return last estimate if not converged
    return q_guess, f_eff


def propagate_v2(optics, z_points, w0, wavelength, P):

    q = q_at_waist(w0, wavelength)
    z_current = 0
    w_z = np.zeros_like(z_points)

    # Sort optics by position
    optics_sorted = sorted(optics, key=lambda o: o['z'])

    # Make a working copy so original list is not modified
    optics_work = [dict(o) for o in optics_sorted]

    for i, z in enumerate(z_points):

        # Process all optics up to this z
        while len(optics_work) > 0 and optics_work[0]['z'] <= z:

            elem = optics_work.pop(0)
            z_elem = elem['z']

            # Free-space propagation to element
            L = z_elem - z_current
            if L > 0:
                q = apply_matrix(q, M_free(L))

            z_current = z_elem

            q, f_eff = apply_thermal_lens(
                q_in=q,
                wavelength=wavelength,
                f_base=elem.get('f_base'),
                m0=elem.get('m0'),
                P=P
            )

        # Propagate from last optic to current z
        L = z - z_current
        q_temp = apply_matrix(q, M_free(L))
        w_z[i] = waist_from_q(q_temp, wavelength)

    return w_z

#%% Animation
def AnimateBeamAfterLastOptic(
        optics,
        z_plot,              # fixed observation axis
        z_lens_positions,    # lens positions to animate over (frames)
        lens_to_move,
        P_anim,
        w0,
        wavelength=1064e-9,
        z0=0):

    fig, ax = plt.subplots(figsize=(9,5))

    line, = ax.plot(z_plot*1e3, np.zeros_like(z_plot), lw=2)
    lens_marker = ax.axvline(0, color='r', linestyle='--', alpha=0.6)

    ax.set_xlim(z_plot.min()*1e3, z_plot.max()*1e3)
    ax.set_ylim(0, 3000)

    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Beam radius (µm)')
    ax.set_title(f'Beam radius vs z as {lens_to_move} moves')

    text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=12)

    def init():
        line.set_ydata(np.zeros_like(z_plot))
        return line, lens_marker, text

    def update(frame):

        z_new = z_lens_positions[frame]

        # move lens on a fresh copy
        optics_frame = [o.copy() for o in optics]
        for o in optics_frame:
            if o['name'] == lens_to_move:
                o['z'] = z_new

        w_z, _ = propagate(optics_frame, z_plot, w0, wavelength, P_anim, z0=z0)

        line.set_ydata(w_z*1e6)
        lens_marker.set_xdata([z_new*1e3, z_new*1e3])
        text.set_text(f"{lens_to_move} at z = {z_new*1e3:.2f} mm")

        return line, lens_marker, text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(z_lens_positions),
        init_func=init,
        interval=30,
        blit=True
    )

    plt.tight_layout()
    plt.show()

    return ani


def AnimateBeamVsPower(
        optics,
        z_plot,          # fixed observation axis
        P_values,        # powers to animate over (frames)
        w0,
        wavelength=1064e-9,
        z0=0):
    
    plt.rcParams['font.size'] =13

    fig, ax = plt.subplots(figsize=(9,5))

    line, = ax.plot(z_plot*1e3, np.zeros_like(z_plot), lw=2)

    ax.set_xlim(z_plot.min()*1e3, z_plot.max()*1e3)
    ax.set_ylim(0, 3000)

    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Beam radius (µm)')

    text = ax.text(0.72, 0.9, '', transform=ax.transAxes, fontsize=12)

    # lens position markers
    for optic in optics:
        if optic.get('f_base') is not None or optic.get('m0') is not None:
            z_lens = optic['z'] * 1e3
            ax.axvline(z_lens, linestyle=':', alpha=0.7)
            ax.text(z_lens, 2800, optic['name'],
                    rotation=90,
                    verticalalignment='top',
                    horizontalalignment='center',
                    fontsize=10)

    def init():
        line.set_ydata(np.zeros_like(z_plot))
        return line, text

    def update(frame):

        P = P_values[frame]

        # Fresh optics copy (propagate destroys the list)
        optics_frame = [o.copy() for o in optics]

        w_z, _ = propagate(optics_frame, z_plot, w0, wavelength, P, z0=z0)

        line.set_ydata(w_z*1e6)
        text.set_text(f"Power = {P:.2f} W")

        return line, text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(P_values),
        init_func=init,
        interval=40,
        blit=True
    )

    plt.tight_layout()
    plt.show()

    return ani

def AnimateBeamAndFocusVsPower(
        optics,
        z_plot,
        P_values,
        w0,
        wavelength=1064e-9,
        z0=0):

    plt.rcParams['font.size'] = 13

    print("Precomputing frames (fast exact method)...")

    # Precompute beam profiles and exact waist positions
    w_frames = []
    z_focus_list = []

    z_last = max(o['z'] for o in optics)

    for P in P_values:
        optics_copy = [o.copy() for o in optics]

        # Full beam profile
        w_z, _ = propagate(optics_copy, z_plot, w0, wavelength, P, z0=z0)
        w_frames.append(w_z * 1e6)

        # Exact waist location from q (no search!)
        q_last = q_after_last_optic(optics, w0, wavelength, P, z0=z0)
        z_waist = -np.real(q_last) + z_last
        z_focus_list.append(z_waist * 1e3)

    w_frames = np.array(w_frames)
    z_focus_list = np.array(z_focus_list)

    print("Precompute complete.")

    # ------------------------------------------------------------
    # Figure
    fig, (ax_beam, ax_focus) = plt.subplots(
        1, 2, figsize=(13,5),
        gridspec_kw={'width_ratios': [3, 1]}
    )

    # Left panel
    beam_line, = ax_beam.plot(z_plot*1e3, w_frames[0], lw=2)
    power_text = ax_beam.text(0.02, 0.92, '',
                              transform=ax_beam.transAxes)

    ax_beam.set_xlim(z_plot.min()*1e3, z_plot.max()*1e3)
    ax_beam.set_ylim(0, 1.1*np.max(w_frames))
    ax_beam.set_xlabel('z (mm)')
    ax_beam.set_ylabel('Beam radius (µm)')
    ax_beam.set_title('Beam radius vs z')

    for optic in optics:
        z_lens = optic['z'] * 1e3
        ax_beam.axvline(z_lens, linestyle=':', alpha=0.6)
        ax_beam.text(z_lens, ax_beam.get_ylim()[1]*0.95,
                     optic['name'],
                     rotation=90, va='top', ha='center', fontsize=9)

    # Right panel
    ax_focus.set_xlim(P_values.min(), P_values.max())
    ax_focus.set_ylim(z_focus_list.min()*0.98,
                      z_focus_list.max()*1.02)

    ax_focus.set_xlabel('Power (W)')
    ax_focus.set_ylabel('Focus after telescope (mm)')
    ax_focus.set_title('Focus position vs Power')

    focus_line, = ax_focus.plot([], [], lw=2)
    focus_point, = ax_focus.plot([], [], 'o')

    # Animation
    def init():
        beam_line.set_ydata(w_frames[0])
        focus_line.set_data([], [])
        focus_point.set_data([], [])
        return beam_line, focus_line, focus_point, power_text

    def update(i):
        beam_line.set_ydata(w_frames[i])
        power_text.set_text(f'Power = {P_values[i]:.2f} W')

        focus_line.set_data(P_values[:i+1],
                            z_focus_list[:i+1])
        focus_point.set_data(P_values[i],
                             z_focus_list[i])

        return beam_line, focus_line, focus_point, power_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(P_values),
        init_func=init,
        interval=40,
        blit=True
    )

    plt.tight_layout()
    plt.show()

    return ani

# Animate beam radius vs z for multiple input waists (z0) simultaneously.
def AnimateBeamVsPowerMultipleZ0(
        optics,
        z_plot,
        P_values,
        w0,
        z0_list,            
        wavelength=1064e-9
    ):
    
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import numpy as np

    plt.rcParams['font.size'] = 13

    # Precompute beam profiles
    print("Precomputing frames for all z0 values...")
    w_frames = []  # shape: (num_z0, num_P, len(z_plot))
    for z0 in z0_list:
        w_curves = []
        for P in P_values:
            optics_copy = [o.copy() for o in optics]
            w_z, _ = propagate(optics_copy, z_plot, w0, wavelength, P, z0=z0)
            w_curves.append(w_z*1e6)  # convert to µm
        w_frames.append(np.array(w_curves))
    w_frames = np.array(w_frames)  # shape (num_z0, num_P, len(z_plot))
    print("Precompute complete.")

    # ------------------- Figure setup -------------------
    fig, ax = plt.subplots(figsize=(9,5))
    colors = plt.cm.cividis(np.linspace(0,1,len(z0_list)))
    lines = []
    for i, z0 in enumerate(z0_list):
        line, = ax.plot(z_plot*1e3, w_frames[i,0], lw=2, color=colors[i], label=f'z0={z0*1e3:.1f} mm')
        lines.append(line)

    ax.set_xlim(z_plot.min()*1e3, z_plot.max()*1e3)
    ax.set_ylim(0, 1.1*np.max(w_frames))
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Beam radius (µm)')
    ax.set_title('Beam radius vs z for multiple input waists')
    ax.legend()
    ax.grid(True, alpha=0.3)

    power_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=12)

    # ------------------- Animation -------------------
    def init():
        for i, line in enumerate(lines):
            line.set_ydata(w_frames[i,0])
        power_text.set_text('')
        return lines + [power_text]

    def update(frame):
        P = P_values[frame]
        for i, line in enumerate(lines):
            line.set_ydata(w_frames[i, frame])
        power_text.set_text(f'Power = {P:.2f} W')
        return lines + [power_text]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(P_values),
        init_func=init,
        interval=50,
        blit=True
    )

    plt.tight_layout()
    plt.show()

    return ani