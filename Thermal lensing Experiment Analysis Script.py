import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from ImageAnalysis import ImageAnalysisCode
import ThermalLensExperimentLibrary as TLE

plt.close('all')

# dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
dataRootFolder = r'C:/Users/wmmax/Documents/Lehigh/Sommer Group/Experiment Data'
date = '3/5/2026'

camera = 'Basler'
powr = [15,30,50,70]
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
    
    # 3.3.2026 -- Lens only
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
    data_folder.append(fr'{camera}/Lens and WP more distance BSPM 267.4 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP more distance BSPM 275.06 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP more distance BSPM 281.82 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP more distance BSPM 289.58 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP more distance BSPM 296.6 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP more distance BSPM 304 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP more distance BSPM 310.6 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP more distance BSPM 318 mm power {p}')
    
    # 3.6.2026 -- Lens, WP, & Polarizer
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 266.49 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 274.12 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 281.21 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 288.9 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 294.83 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 302.4 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 310.06 mm power {p}')
    # data_folder.append(fr'{camera}/Lens WP and Polarizer BSPM 317.69 mm power {p}')


rep = 6
commonPhrase = True
save = False

angle = 0

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

if save:
    results.to_csv('Focus shift SPX059AR.1 300 mm lens - Lens only.csv', index=False)

#%%

# TLE.Plot_QuantvsPower('z0_X', results)
# TLE.Plot_QuantvsPower('z0_Y', results)
# TLE.Plot_QuantvsPower('w0_X', results)
# TLE.Plot_QuantvsPower('w0_Y', results)

#%%
scale=1e3

plt.figure(figsize=(4.5,3.5))

P_W = 2.34*results['Power'] - 30.1
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

P_W = 2.34*results['Power'] - 30.1
plt.errorbar(P_W, results['w0_X fit']*scale, yerr=results['w0_X fit err']*scale, fmt='o-', capsize=3)
plt.errorbar(P_W, results['w0_Y fit']*scale, yerr=results['w0_Y fit err']*scale, fmt='o-', capsize=3)

plt.title('w0 shift vs power', fontsize=14)
plt.xlabel('Power (W)')
plt.ylabel('μm')
plt.grid(True, alpha=0.3)
plt.legend(['$w_{0X}$', '$w_{0Y}$'])
plt.tight_layout()

    
#%%

for p in powr:
    power_folders = [folder for folder in dataPath if f'power {p}' in folder]
    power_stats = stats[stats['Power'] == p].sort_values(by='Distance')
    
    TLE.Plot_BeamEvolutionXYWithImages(
        power_stats, 
        power_folders, 
        power=p, 
        camera=camera, 
        ROI=ROI,
        crop_window=300
    )
