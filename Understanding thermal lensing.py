import numpy as np
import matplotlib.pyplot as plt

m0 = -4e-9
w = 0.25e-3
f0 = -np.array([125e-3, 250e-3, 350e-3, 500e-3])
# f0 = np.array([500e-3])
# f0 = np.array([-125e-3, -250e-3])

def thermalF(P,w,m0):
    return w**2 / (m0*P)

def effectiveF(f0, fth):
    f = 1/(1/f0 + 1/fth)
    return f

P = np.linspace(0,150,500)

fth = thermalF(P, w, m0)

plt.rcParams.update({'font.size': 12})
plt.figure()
for j in f0:
    effectiveFocalLength = effectiveF(j, fth)
    
    # mask = np.abs(effectiveFocalLength) < 50000e-3
    # Pfilt = P[mask]
    # EFLfilt = effectiveFocalLength[mask]

    # plt.plot(Pfilt, 1/(EFLfilt*1e3), label=f'f0 = {j*1e3} mm')
    plt.plot(P, 1/(effectiveFocalLength*1e3), label=f'f0 = {j*1e3} mm')


plt.xlabel('Power (W)')
plt.ylabel('1/$f_{eff}$ (1/mm)')
plt.legend()
plt.grid(True, alpha=0.5)
plt.title(f'Effective focusing power ($m_0$ = {m0})', fontsize=14)
plt.tight_layout()

#%%
plt.figure()
ftest = np.linspace(75e-3, 500e-3, 200)
wtest = np.array([0.25e-3, 0.5e-3, 1e-3, 1.5e-3])

for waist in wtest:
    critPow = -waist**2/(ftest*m0)
    plt.plot(ftest*1e3,critPow, label=f'w = {waist*1e3} mm')
    
plt.xlabel('$f_0$ (mm)')
plt.ylabel('Critical power (W)')
plt.legend()
plt.title('Critical power vs. nominal focal length')
plt.grid(True)
plt.yscale('log')
plt.ylim(ymin=20)
