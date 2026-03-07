import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from ImageAnalysis import ImageAnalysisCode
import ThermalLensExperimentLibrary as TLE

plt.close('all')

# dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
dataRootFolder = r'C:/Users/wmmax/Documents/Lehigh/Sommer Group/Experiment Data'
date = '3/3/2026'

camera = 'Basler'
powr = [15,30,40,50,60,70]
data_folder = []

for p in powr:
    
    # 3.3.2026 -- Lens only
    data_folder.append(fr'{camera}/Lens only BSPM 268.38 mm power {p}')
    data_folder.append(fr'{camera}/Lens only BSPM 276.14 mm power {p}')
    data_folder.append(fr'{camera}/Lens only BSPM 283.17 mm power {p}')
    data_folder.append(fr'{camera}/Lens only BSPM 290.25 mm power {p}')
    data_folder.append(fr'{camera}/Lens only BSPM 297.74 mm power {p}')
    data_folder.append(fr'{camera}/Lens only BSPM 304.88 mm power {p}')
    data_folder.append(fr'{camera}/Lens only BSPM 314.28 mm power {p}')
    data_folder.append(fr'{camera}/Lens only BSPM 321.62 mm power {p}')

    
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

df_Lens = pd.read_csv('Focus shift SPX059AR.1 300 mm lens - Lens only.csv')
df_LensWP = pd.read_csv('Focus shift SPX059AR.1 300 mm lens - Lens and WP.csv')
df_LensWP_moredist = pd.read_csv('Focus shift SPX059AR.1 300 mm lens - Lens and WP more distance.csv')
df_LensWPPolarizer = pd.read_csv('Focus shift SPX059AR.1 300 mm lens - Lens WP and Polarizer.csv')

axis = 'Y'

plt.figure(figsize=(4.5, 3.5))
plt.title(f'z0_{axis} shifts')
plt.xlabel('Power (W)')
plt.ylabel('mm')

for d in [df_Lens, df_LensWP, df_LensWP_moredist, df_LensWPPolarizer]:
    
    plt.errorbar(P_W, d[f'z0_{axis} fit']*1e3, yerr=d[f'z0_{axis} fit err']*1e3, fmt='-o', capsize=3)
       
plt.legend(['Lens', 'Lens & WP', 'Lens & WP (more dist)', 'Lens, WP, & Polarizer'])
plt.grid(True, alpha=0.3)
plt.tight_layout()



plt.figure(figsize=(4.5, 3.5))

plt.title(f'w0_{axis} shifts')
plt.xlabel('Power (W)')
plt.ylabel('μm')

for d in [df_Lens, df_LensWP, df_LensWP_moredist, df_LensWPPolarizer]:
    
    plt.errorbar(P_W, d[f'w0_{axis} fit']*1e6, yerr=d[f'w0_{axis} fit err']*1e6, fmt='-o', capsize=3)
    
    
plt.legend(['Lens', 'Lens & WP', 'Lens & WP (more dist)', 'Lens, WP, & Polarizer'])
plt.grid(True, alpha=0.3)
plt.tight_layout()