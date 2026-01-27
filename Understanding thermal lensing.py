import numpy as np
import matplotlib.pyplot as plt
import ThermalLensLibrary as TL

plt.close('all')

m0 = -4e-9
w0 = 1e-3
f0 = -np.array([125e-3, 250e-3, 350e-3, 500e-3])
# f0 = np.array([-500e-3])
# f0 = np.array([-125e-3, -250e-3])

def thermalF(P,w,m0):
    return w**2 / (m0*P)

def effectiveF(f0, fth):
    f = 1/(1/f0 + 1/fth)
    return f

P = np.linspace(1,2000,500)

fth = thermalF(P, w0, m0)

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1,2,figsize=(8,3.5))
fig.suptitle('Ray analysis')

for j in f0:
    effectiveFocalLength = effectiveF(j, fth)
    
    mask = np.abs(effectiveFocalLength) < 2000e-3
    Pfilt = P[mask]
    EFLfilt = effectiveFocalLength[mask]

    ax[0].plot(Pfilt, EFLfilt*1e3, label=f'f0 = {j*1e3} mm')
    # plt.plot(P, 1/(effectiveFocalLength*1e3), label=f'f0 = {j*1e3} mm')
    # ax[0].plot(P, effectiveFocalLength*1e3, label=f'$f_0$={j*1e3:.0f} mm')
    ax[1].plot(P, 1/(effectiveFocalLength*1e3), label=f'$f_0$={j*1e3:.0f} mm')


ax[0].set_xlabel('Power (W)')
ax[0].set_ylabel('Effective F (mm)')
ax[0].grid(True, alpha=0.3)

ax[1].set_xlabel('Power (W)')
ax[1].set_ylabel('Effective 1/F (1/mm)')
ax[1].grid(True, alpha=0.3)
ax[1].legend(fontsize=10)

fig.text(
    0.5, 0.87,                    # middle, near bottom of the FIGURE
    f'$w_0$={w0*1e3:.0f} mm, $m_0$={m0} m/W',
    ha='center',                  # horizontal alignment
    va='bottom',                  # vertical alignment
    fontsize=12
)

plt.tight_layout()

#%%
plt.figure(figsize=(5,4))
ftest = np.linspace(-75e-3, -500e-3, 200)
wtest = np.array([0.25e-3, 0.5e-3, 1e-3, 1.5e-3])

for waist in wtest:
    critPow = -waist**2/(ftest*m0)
    plt.plot(ftest*1e3,critPow, label=f'w = {waist*1e3} mm')
    
plt.xlabel('$f$ (mm)')
plt.ylabel('Critical power (W)')
plt.legend()
plt.title('P* vs. nominal f')
plt.grid(True)
plt.yscale('log')
plt.ylim(ymin=20)
plt.tight_layout()

#%%

def beam_afterLens(w0,F,wavelength=1064e-9):
    zR = TL.z_R(w0, wavelength)
    z0_after = F*zR**2 / (F**2 + zR**2)
    # w0_after = w0*F / np.sqrt(F**2 + zR**2)
    w0_after = w0 / np.sqrt(1+zR**2 / F**2)
    return z0_after, w0_after


fig,ax = plt.subplots(1,2, figsize=(8,3.5))
fig.suptitle('Gaussian beam analysis')

for j in f0:
    F1 = effectiveF(j, fth)
    z0_after, w0_after = beam_afterLens(w0, F1)
    
    ax[0].plot(P, z0_after*1e3, label=f'f0 = {j*1e3} mm')
    ax[1].plot(P, w0_after*1e6, label=f'f0 = {j*1e3} mm')

ax[0].set_xlabel('Power (W)')
ax[0].set_ylabel('Focus after lens (mm)')
ax[0].grid(True, alpha=0.3)
ax[0].legend(fontsize=10)


ax[1].set_xlabel('Power (W)')
ax[1].set_ylabel('Focused waist (um)')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()