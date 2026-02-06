import numpy as np
import matplotlib.pyplot as plt
import ThermalLensLibrary as TL

plt.close('all')
plt.rcParams.update({'font.size': 12})


m0 = 4e-9
w0 = 1e-3
zR_num = TL.z_R(w0)

z0 = np.array([-3*zR_num, -1*zR_num, 0*zR_num, 1*zR_num, 3*zR_num])
f0 = np.array([125e-3, 250e-3, 350e-3, 500e-3])

P = np.linspace(1,10000,500)  
    
TL.Plot_SingleLensAnalysis(P, w0, m0,
    sweep_param='z0',
    sweep_values=z0,
    fixed_value=-100e-3,
    focus_scale='mm',
    F1_scale=True, 
    delta_focus=True
)

#%% Diagnostics

z0_cont = np.linspace(-4*zR_num, 4*zR_num, 100)
wL1_cont = TL.waist_L1(w0, z0_cont)
P_ref = 100

alpha_100mm = m0 * (-100e-3) / wL1_cont**2
alpha_500mm = m0 * (-500e-3) / wL1_cont**2

thermalF = wL1_cont**2 / (m0*P_ref)
eff_100mm = TL.effective_focalLength(-100e-3, P_ref, m0, wL1_cont)
eff_500mm = TL.effective_focalLength(-500e-3, P_ref, m0, wL1_cont)

plt.figure(figsize=(5,4))
plt.plot(z0_cont/zR_num, alpha_100mm, label='$f_0$=-100 mm')
plt.plot(z0_cont/zR_num, alpha_500mm, label='$f_0$=-500 mm')
plt.xlabel('$z_0/z_R$')
plt.ylabel(r'$\alpha$ (1/W)')
plt.legend(); plt.tight_layout()

plt.figure(figsize=(5,4))
plt.plot(z0_cont/zR_num, -1/alpha_100mm, label='$f_0$=-100 mm')
plt.plot(z0_cont/zR_num, -1/alpha_500mm, label='$f_0$=-500 mm')
plt.xlabel('$z_0/z_R$')
plt.ylabel('P* (W)')
plt.grid(True, alpha=0.3); plt.yscale('log')
plt.legend(); plt.tight_layout()

plt.figure(figsize=(5,4))
plt.title(f'Effective F vs. $z_0/z_R$, fixed P={P_ref:.0f} W')
plt.plot(z0_cont/zR_num, eff_100mm*1e3 / 100, label='$f_0$=-100 mm')
plt.plot(z0_cont/zR_num, eff_500mm*1e3 / 500, label='$f_0$=-500 mm')
plt.xlabel('$z_0/z_R$')
plt.ylabel('$F_1 / f_0$')
plt.legend(); plt.tight_layout()


#%% Ray Analysis

fig, ax = plt.subplots(1,2,figsize=(8,3.5))
fig.suptitle('Ray analysis')

for j in f0:
    effectiveFocalLength = TL.effective_focalLength(j, P, m0, w0)
    
    # mask = np.abs(effectiveFocalLength) < 2000e-3
    # Pfilt = P[mask]
    # EFLfilt = effectiveFocalLength[mask]

    # ax[0].plot(Pfilt, EFLfilt*1e3, label=f'f0 = {j*1e3} mm')
    # plt.plot(P, 1/(effectiveFocalLength*1e3), label=f'f0 = {j*1e3} mm')
    ax[0].plot(P, effectiveFocalLength*1e3, label=f'$f_0$={j*1e3:.0f} mm')
    ax[1].plot(P, 1/(effectiveFocalLength*1e3), label=f'$f_0$={j*1e3:.0f} mm')


ax[0].set_xlabel('Power (W)')
ax[0].set_ylabel('Effective F (mm)')
ax[0].grid(True, alpha=0.3)

ax[1].set_xlabel('Power (W)')
ax[1].set_ylabel('Effective 1/F (1/mm)')
ax[1].grid(True, alpha=0.3)
ax[1].legend(fontsize=10)

fig.text(
    0.5, 0.87,                    
    f'$w_0$={w0*1e3:.0f} mm, $m_0$={m0} m/W',
    ha='center',
    va='bottom',
    fontsize=12
)

plt.tight_layout()

#%% Critical Power
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