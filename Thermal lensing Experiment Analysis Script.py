import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from ImageAnalysis import ImageAnalysisCode
import ThermalLensExperimentLibrary as TLE

plt.close('all')

# dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
dataRootFolder = r'C:/Users/wmmax/Documents/Lehigh/Sommer Group/Experiment Data'
date = '3/30/2026'
# date='12/1/2025'

camera = 'Basler'
powr = [15,30,40,50,60,70]
# powr = [15,30,50,70]
# powr = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
data_folder = []

for p in powr:
    
    # 1.20.2026 -- first pass lens
    # data_folder.append(fr'{camera}/After first pass 195 mm power {p}')
    # data_folder.append(fr'{camera}/After first pass 202 mm power {p}')
    # data_folder.append(fr'{camera}/After first pass 205 mm power {p}')
    # data_folder.append(fr'{camera}/After first pass 208 mm power {p}')
    # data_folder.append(fr'{camera}/After first pass 213 mm power {p}')
    # data_folder.append(fr'{camera}/After first pass 216 mm power {p}')
    # data_folder.append(fr'{camera}/After first pass 219 mm power {p}')
    # data_folder.append(fr'{camera}/After first pass 226 mm power {p}')
    # data_folder.append(fr'{camera}/After first pass 231 mm power {p}')
    # data_folder.append(fr'{camera}/After first pass 237 mm power {p}')
    
    # 1.21.2026 -- second pass lens
    # data_folder.append(fr'{camera}/After Second pass 312 mm power {p}')
    # data_folder.append(fr'{camera}/After Second pass 319 mm power {p}')
    # data_folder.append(fr'{camera}/After Second pass 325 mm power {p}')
    # data_folder.append(fr'{camera}/After Second pass 326 mm power {p}')
    # data_folder.append(fr'{camera}/After Second pass 330 mm power {p}')
    # data_folder.append(fr'{camera}/After Second pass 331 mm power {p}')
    # data_folder.append(fr'{camera}/After Second pass 336 mm power {p}')
    
    # 11.13.2025 -- 300 mm lens (SPX059AR.1)
    # data_folder.append(fr'{camera}/SPX059AR.1 277 mm power {p}')
    # data_folder.append(fr'{camera}/SPX059AR.1 283 mm power {p}')
    # data_folder.append(fr'{camera}/SPX059AR.1 290 mm power {p}')
    # data_folder.append(fr'{camera}/SPX059AR.1 297 mm power {p}')
    # data_folder.append(fr'{camera}/SPX059AR.1 301 mm power {p}')
    # data_folder.append(fr'{camera}/SPX059AR.1 307 mm power {p}')
    # data_folder.append(fr'{camera}/SPX059AR.1 315 mm power {p}')
    
    # 11.14.2025 -- 300 mm lens again (SPX059AR.1)
    # data_folder.append(fr'{camera}/SPX059AR.1 again 285 mm power {p}')
    # data_folder.append(fr'{camera}/SPX059AR.1 again 292 mm power {p}')
    # data_folder.append(fr'{camera}/SPX059AR.1 again 299 mm power {p}')
    # data_folder.append(fr'{camera}/SPX059AR.1 again 311 mm power {p}')
    # data_folder.append(fr'{camera}/SPX059AR.1 again 317 mm power {p}')
    
    # 3.3.2026 -- 300 mm lens only (SPX059AR.1)
    # data_folder.append(fr'{camera}/Lens only BSPM 268.38 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only BSPM 276.14 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only BSPM 283.17 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only BSPM 290.25 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only BSPM 297.74 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only BSPM 304.88 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only BSPM 314.28 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only BSPM 321.62 mm power {p}')

    
    # 3.4.2026 -- Lens & WP, short distance
    # data_folder.append(fr'{camera}/Lens and WP BPSM 268.73 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP BPSM 276.01 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP BPSM 284.41 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP BPSM 292.15 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP BPSM 299.64 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP BPSM 307.24 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP BPSM 314.28 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP BPSM 321.62 mm power {p}')
    
    # 3.5.2026 -- Lens & WP, more distance
    # data_folder.append(fr'{camera}/Lens and WP more distance BSPM 267.4 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP more distance BSPM 275.06 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP more distance BSPM 281.82 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP more distance BSPM 289.58 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP more distance BSPM 296.6 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP more distance BSPM 304 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP more distance BSPM 310.6 mm power {p}')
    # data_folder.append(fr'{camera}/Lens and WP more distance BSPM 318 mm power {p}')
    
    # 3.6.2026 -- Lens, WP, & Polarizer
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 266.49 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 274.12 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 281.21 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 288.9 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 294.83 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 302.4 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 310.06 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 317.69 mm power {p}')
    
    # 12.1.2025 -- 125 mm lens (SPX023AR.1)
    # data_folder.append(fr'{camera}/SPX023AR.1 110 mm power {p}')
    # data_folder.append(fr'{camera}/SPX023AR.1 113 mm power {p}')
    # data_folder.append(fr'{camera}/SPX023AR.1 117 mm power {p}')
    # data_folder.append(fr'{camera}/SPX023AR.1 121 mm power {p}')
    # data_folder.append(fr'{camera}/SPX023AR.1 124 mm power {p}')
    # data_folder.append(fr'{camera}/SPX023AR.1 128 mm power {p}')
    # data_folder.append(fr'{camera}/SPX023AR.1 132 mm power {p}')
    
    # 3.16.2026 -- 350 mm lens (SPX030AR.33)
    # data_folder.append(fr'{camera}/Lens only SPX030AR.33 BSPM 311.4 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only SPX030AR.33 BSPM 318.3 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only SPX030AR.33 BSPM 325.9 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only SPX030AR.33 BSPM 333.1 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only SPX030AR.33 BSPM 340.7 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only SPX030AR.33 BSPM 348.6 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only SPX030AR.33 BSPM 356.3 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only SPX030AR.33 BSPM 362.5 mm power {p}')
    # data_folder.append(fr'{camera}/Lens only SPX030AR.33 BSPM 370.1 mm power {p}')
    
    # 3.12.2026 -- telescope collimation
    # data_folder.append(fr'{camera}/Last telescope attempt pos1 111.8 mm power {p}')
    # data_folder.append(fr'{camera}/Last telescope attempt pos1 227.8 mm power {p}')
    # data_folder.append(fr'{camera}/Last telescope attempt pos1 389.8 mm power {p}')
    # data_folder.append(fr'{camera}/Last telescope attempt pos1 498.8 mm power {p}')

    # data_folder.append(fr'{camera}/Last telescope attempt pos2 112.8 mm power {p}')
    # data_folder.append(fr'{camera}/Last telescope attempt pos2 228.8 mm power {p}')
    # data_folder.append(fr'{camera}/Last telescope attempt pos2 390.8 mm power {p}')
    # data_folder.append(fr'{camera}/Last telescope attempt pos2 499.8 mm power {p}')
    
    # 3.26.2026 -- first order propagation
    # data_folder.append(fr'{camera}/First order BSPM 140.4 mm power {p}')
    # data_folder.append(fr'{camera}/First order BSPM 319.3 mm power {p}')
    # data_folder.append(fr'{camera}/First order BSPM 496.3 mm power {p}')
    # data_folder.append(fr'{camera}/First order BSPM 664.8 mm power {p}')
    # data_folder.append(fr'{camera}/First order NEXT DAY BSPM 880 mm power {p}')

    
    # 3.27.2026 -- focus first order beam
    # data_folder.append(fr'{camera}/Focus first order BSPM 292.4 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order BSPM 297 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order BSPM 304.3 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order BSPM 312.5 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order BSPM 320 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order BSPM 326.6 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order BSPM 334.2 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order BSPM 342.2 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order BSPM 349.6 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order BSPM 356.2 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order BSPM 363.5 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order BSPM 370.3 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order 377.7 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order 384.5 mm power {p}')
    # data_folder.append(fr'{camera}/Focus first order 392.3 mm power {p}')
    
    # 3.30.2026 -- 175 mm lens
    data_folder.append(fr'{camera}/Focus 175lens BSPM 150.5 mm power {p}')
    data_folder.append(fr'{camera}/Focus 175lens BSPM 157.4 mm power {p}')
    data_folder.append(fr'{camera}/Focus 175lens BSPM 165.1 mm power {p}')
    data_folder.append(fr'{camera}/Focus 175lens BSPM 173.6 mm power {p}')
    data_folder.append(fr'{camera}/Focus 175lens BSPM 181.2 mm power {p}')
    data_folder.append(fr'{camera}/Focus 175lens BSPM 189 mm power {p}')
    data_folder.append(fr'{camera}/Focus 175lens BSPM 196.6 mm power {p}')







rep = 6
commonPhrase = True
save = False

angle = 5

# rowstart=720
# rowend=880
# columnstart=175
# columnend=525
rowstart=1
rowend=-1
columnstart=1
columnend=-1
ROI = [rowstart, rowend, columnstart, columnend]

dayFolder = ImageAnalysisCode.GetDataLocation(date, dataRootFolder)
dataPath = [ os.path.join(dayFolder, j) for j in data_folder]

#%%
df = TLE.Fit_GaussianRawImages(dataPath, camera, ROI, repetition=rep, angle=angle, doPlot=False, commonPhrase=True)

colsForAnalysis = ['Xwidth', 'Ywidth']

stats = TLE.RawFitStats(df, colsForAnalysis)

results = TLE.Fit_GaussianBeamRadius(stats, colsForAnalysis, doPlot=True)

# if save:
    # stats.to_csv('stats_collimationRetry_03122026.csv', index=False)
    # results.to_csv('Focus shift SPX059AR.1 300 mm lens - Lens only.csv', index=False)

#%%

# TLE.Plot_QuantvsPower('z0_X', results)
# TLE.Plot_QuantvsPower('z0_Y', results)
# TLE.Plot_QuantvsPower('w0_X', results)
# TLE.Plot_QuantvsPower('w0_Y', results)

#%%

# P_W = np.array([1.7, 14.75, 34.06, 52.75, 67.28, 76.2, 81.5, 84.3])
P_W = 2.34*results['Power'] - 30.1

scale=1e3

plt.figure(figsize=(4.5,3.5))

plt.errorbar(P_W, results['z0_X fit']*scale, yerr=results['z0_X fit err']*scale, fmt='o-', capsize=3)
plt.errorbar(P_W, results['z0_Y fit']*scale, yerr=results['z0_Y fit err']*scale, fmt='o-', capsize=3)

plt.title('z0 shift vs power', fontsize=14)
plt.xlabel('Power (W)')
plt.ylabel('mm')
plt.grid(True, alpha=0.3)
plt.legend(['$z_{0X}$', '$z_{0Y}$'])
plt.tight_layout()

#%%
scale=1e6

plt.figure(figsize=(4.5,3.5))

plt.errorbar(P_W, results['w0_X fit']*scale, yerr=results['w0_X fit err']*scale, fmt='o-', capsize=3)
plt.errorbar(P_W, results['w0_Y fit']*scale, yerr=results['w0_Y fit err']*scale, fmt='o-', capsize=3)

plt.title('w0 shift vs power', fontsize=14)
plt.xlabel('Power (W)')
plt.ylabel('μm')
plt.grid(True, alpha=0.3)
plt.legend(['$w_{0X}$', '$w_{0Y}$'])
plt.tight_layout()

    
#%% Plot beam images with propagation graph

# for p in powr:
#     power_folders = [folder for folder in dataPath if f'power {p}' in folder]
#     power_stats = stats[stats['Power'] == p].sort_values(by='Distance')
    
#     TLE.Plot_BeamEvolutionXYWithImages(
#         power_stats, 
#         power_folders, 
#         power=p, 
#         camera=camera, 
#         ROI=ROI,
#         crop_window=300
#     )

#%% Extract m0 with known w0, z0, and f
from scipy.optimize import curve_fit

z0 = 0
lamb = 1064e-9 

P_data = 2.34*results['Power'].values - 30.1
# P_data = np.array([1.7, 14.75, 34.06, 52.75, 67.28, 76.2, 81.5, 84.3])

var2analyze = ['X', 'Y']

plt.figure(figsize=(5, 4))
plt.rcParams['font.size']=12

for j in range(len(var2analyze)):
    
    
    if var2analyze == 'X':
        w0_meas = 1.95e-3
        w0_meas_err = 0.06e-3
    else:
        w0_meas = 2.14e-3
        w0_meas_err = 0.08e-3
    
    z_R = (np.pi * w0_meas**2) / lamb


    z0_data = results['z0_'+ var2analyze[j] +' fit'].values
    z0_data_err = results['z0_'+ var2analyze[j] +' fit err']
    
    f_measured = z0_data[0]       
    f_measured_err = z0_data_err[0]


    def FocusAfterLens(P, alpha):
        F = f_measured / (1 + alpha * P)
        z0_prime = F * (z_R**2 - z0 * (F - z0)) / ((F - z0)**2 + z_R**2)
        return z0_prime


    initial_guess = [1e-4] 
    
    popt, pcov = curve_fit(FocusAfterLens, P_data, z0_data, p0=initial_guess)
    
    alpha_fit = popt[0]
    cov_aa = pcov[0][0]
    alpha_err = np.sqrt(cov_aa)
    
    m0 = alpha_fit * w0_meas**2 / f_measured
    
    # derivatives
    dm0_da = w0_meas**2 / f_measured
    dm0_dw = 2*alpha_fit*w0_meas / f_measured
    dm0_df = -alpha_fit * w0_meas**2 / f_measured**2
    
    # variance terms for each parameter
    var_a = (dm0_da * alpha_err)**2
    var_w = (dm0_dw * w0_meas_err)**2
    var_f = (dm0_df * f_measured_err)**2
    
    # Total error in m_0
    m0_err = np.sqrt(var_a + var_f + var_w)
    
    print('------- '+var2analyze[j]+' fit values -------')
    print(f"alpha: {alpha_fit:.3e} +/- {alpha_err:.3e} W^-1")
    print(f"m0: {m0:.3e} +/- {m0_err:.3e} m/W\n")

    P_fit_curve = np.linspace(min(P_data), max(P_data), 1000)
    z0_fit_curve = FocusAfterLens(P_fit_curve, alpha_fit)

    plt.errorbar(P_data, z0_data*1e3, yerr=z0_data_err*1e3, 
                 fmt = 'o',
                 label='z0'+var2analyze[j]+' data', 
                 color='C'+str(j+2),
                 capsize=3
                 )
    
    plt.plot(P_fit_curve, z0_fit_curve*1e3, 
             label=f'$\\alpha$={alpha_fit:.2e} $\pm$ {alpha_err:.2e} 1/W', 
             color='C'+str(j+2)
             )

plt.xlabel('Power (W)')
plt.ylabel('$z_0^\prime$ (mm)')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

#%% include M^2 factor in zR

plt.figure(figsize=(6, 5))
plt.rcParams['font.size'] = 12

for j, axis in enumerate(var2analyze):
    
    if axis == 'X':
        w0_meas = 1.95e-3
        w0_meas_err = 0.06e-3
    else:
        w0_meas = 2.14e-3
        w0_meas_err = 0.08e-3
    
    z0_data = results['z0_'+ axis +' fit'].values
    z0_data_err = results['z0_'+ axis +' fit err'].values
    
    f_measured = z0_data[0]       
    f_measured_err = z0_data_err[0]

    # include M^2 as a fit parameter
    def FocusAfterLens(P, alpha, M2):
        z_R_fit = (np.pi * w0_meas**2) / (M2 * lamb)
        
        F = f_measured / (1 + alpha * P)
        z0_prime = F * (z_R_fit**2 - z0 * (F - z0)) / ((F - z0)**2 + z_R_fit**2)
        return z0_prime

    initial_guess = [1e-4, 1] 
    
    popt, pcov = curve_fit(FocusAfterLens, P_data, z0_data, p0=initial_guess)
    
    alpha_fit, M2_fit = popt
    alpha_err, M2_err = np.sqrt(np.diag(pcov))
    
    # Calculate m0 with the fitted alpha
    m0 = alpha_fit * w0_meas**2 / f_measured
    
    # --- Error Propagation for m0 ---
    dm0_da = w0_meas**2 / f_measured
    dm0_dw = 2 * alpha_fit * w0_meas / f_measured
    dm0_df = -alpha_fit * w0_meas**2 / f_measured**2
    
    var_a = (dm0_da * alpha_err)**2
    var_w = (dm0_dw * w0_meas_err)**2
    var_f = (dm0_df * f_measured_err)**2
    m0_err = np.sqrt(var_a + var_f + var_w)
    
    # Output results
    print(f'------- {axis} fit values -------')
    print(f"alpha: {alpha_fit:.3e} +/- {alpha_err:.3e} W^-1")
    print(f"M^2:   {M2_fit:.3f} +/- {M2_err:.3f}")
    print(f"m0:    {m0:.3e} +/- {m0_err:.3e} m/W\n")

    # Plotting
    P_fit_curve = np.linspace(min(P_data), max(P_data), 1000)
    z0_fit_curve = FocusAfterLens(P_fit_curve, alpha_fit, M2_fit)

    plt.errorbar(P_data, z0_data*1e3, yerr=z0_data_err*1e3, 
                 fmt='o', label=f'Data {axis}', color=f'C{j}', capsize=3)
    
    plt.plot(P_fit_curve, z0_fit_curve*1e3, 
             label=f'Fit {axis} ($M^2$={M2_fit:.2f})', color=f'C{j}')

plt.xlabel('Power (W)')
plt.ylabel('$z_0^\prime$ (mm)')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()