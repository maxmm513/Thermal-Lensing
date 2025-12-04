import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.close('all')

lensType = 'SPX023AR.1'
axis = 'Y'

if lensType == 'SPX030AR.33':
    material = 'UVFS'
    nominal_f_mm = 350
    m0_guess = 4e-9
    
elif lensType == 'SPX031AR.1':
    material = 'HPFS'
    nominal_f_mm = 500
    m0_guess = 1e-9
    
elif lensType == 'SPX059AR.1':
    material = 'HPFS'
    nominal_f_mm = 300
    m0_guess = 1e-9
    
elif lensType == 'SPX023AR.1':
    material = 'HPFS'
    nominal_f_mm = 125
    m0_guess = 1e-9

folder = 'C:/Users\wmmax\Documents\Lehigh\Sommer Group\Thermal lensing'
filename = 'Focus shift data ' + lensType + ' ' + str(nominal_f_mm) + ' mm lens.csv'

data = pd.read_csv(folder+'/'+filename)
dfA = data[data['Axis'] == axis]

# dfA = dfA.drop([2])

P = dfA['Power_W']
z0 = dfA['z0_mm'] * 1e-3
z0_err = dfA['z0_err_mm'] * 1e-3

#%%
f0_meas = z0[1]
w = np.array([1.8e-3, 1.9e-3, 2e-3, 2.1e-3])

alpha_guess = m0_guess*f0_meas/w[0]**2

def effective_f(P, alpha):
    return f0_meas / (1 + alpha*P)


param, pcov = curve_fit(effective_f, P, z0, p0=[alpha_guess], sigma=z0_err)
alpha_fit = param[0]
alpha_fit_err = np.sqrt(pcov[0,0])   # 1-sigma uncertainty

P_fit = np.linspace(0, max(P), 1000)

f_fit = effective_f(P_fit, alpha_fit)
f_upper = effective_f(P_fit, alpha_fit + alpha_fit_err)
f_lower = effective_f(P_fit, alpha_fit - alpha_fit_err)

# calculate m0 based on focal length at P=0
f0_zeroP = effective_f(0, *param)
m0_fit = alpha_fit * w**2 / f0_zeroP


plt.rcParams['font.size'] = 13
plt.figure(figsize=(5,4))

plt.scatter(P, z0*1e3)
plt.plot(P_fit, f_fit*1e3)
plt.fill_between(P_fit,
                 f_lower*1e3,
                 f_upper*1e3,
                 alpha=0.3,
                 )

ax = plt.gca()
ax.text(0.59, 0.84,
        f'Nominal f = {nominal_f_mm} mm\nf0 = {f0_zeroP*1e3:.2f} mm\nα = {param[0]:.3e} 1/W',
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='round',
            alpha=0.8
        )
        )

plt.xlabel('Power (W)')
plt.ylabel('Focus position (mm)')
plt.tight_layout()
plt.grid(True)

#%%
plt.figure(figsize=(5,4))
plt.plot(w*1e3, m0_fit)
plt.xlabel('Waist at lens (mm)', fontsize=13)
plt.ylabel('Extracted m0 (m/W)', fontsize=13)
plt.tight_layout()
plt.grid(True)

