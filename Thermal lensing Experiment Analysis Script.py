import numpy as np
import matplotlib.pyplot as plt
import os
from ImageAnalysis import ImageAnalysisCode
import ThermalLensExperimentLibrary as TLE

plt.close('all')

# dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
dataRootFolder = r'C:/Users/wmmax/Documents/Lehigh/Sommer Group/Experiment Data'
date = '3/4/2026'

camera = 'Basler'
powr = [15,30,40,50,60,70]
data_folder = []

for p in powr:
    data_folder.append(fr'{camera}/Lens and WP BPSM 268.73 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 276.01 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 284.41 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 292.15 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 299.64 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 307.24 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 314.28 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 321.62 mm power {p}')


rep = 6
commonPhrase = True
quantity = 'Distance (mm)'
var2plot = 'Distance'

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

#%%

def Plot_QuantvsPower(quant, results, polarizer=False):
    
    # convert power % to watts
    if polarizer:
        P_W = 2.17*results['Power'] - 28.4
    else:
        P_W = 2.34*results['Power'] - 30.1
        
    if quant[0] == 'z':
        scale = 1e3
        unit = 'mm'
    elif quant[0] == 'w':
        scale = 1e6
        unit = 'μm'
        
    col_vals = quant+' fit'
    col_errs = quant+' fit err'
    
    plt.figure(figsize=(4,3))    
    plt.errorbar(P_W, results[col_vals]*scale, yerr=results[col_errs]*scale, fmt='o', capsize=3)
    
    plt.title(quant, fontsize=14)
    plt.xlabel('Power (W)')
    plt.ylabel(quant + f' ({unit})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
Plot_QuantvsPower('z0_X', results)

