import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from scipy.optimize import curve_fit
from ImageAnalysis import ImageAnalysisCode
from LightSheetAnalysis import LSAnalysisCode
import datetime
import configparser
from PIL import Image
import cv2


plt.close('all')

# dataRootFolder = r"D:\Dropbox (Lehigh University)\Sommer Lab Shared\Data"
dataRootFolder = r'C:/Users/wmmax/Documents/Lehigh/Sommer Group/Experiment Data'
date = '3/4/2026'

camera = 'Basler'
# powr = [15,30,50,70]
powr = [15,30,40,50,60,70]
# camera = 'Andor'
data_folder = [
    ]

for p in powr:
    data_folder.append(fr'{camera}/Lens and WP BPSM 268.73 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 276.01 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 284.41 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 292.15 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 299.64 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 307.24 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 314.28 mm power {p}')
    data_folder.append(fr'{camera}/Lens and WP BPSM 321.62 mm power {p}')


repetition = 6
commonPhrase = True
quantity = 'Distance (mm)'
var2plot = 'Distance'

doPlot = 0
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

if camera == 'Basler':
    pixSize = 2 #um/px
elif camera == 'FLIR':
    pixSize = 3.75 #um/px
elif camera == 'Andor':
    pixSize = 6.5 #um/pix
#%%

df = pd.DataFrame(columns=['File', 'Condition', 'Value', 'Xcenter', 'Ycenter', 'Xwidth', 'Ywidth', 'Xamp', 'Yamp'])

if commonPhrase:

    conditions, values, distances = ImageAnalysisCode.RecognizeCommonPhrase(dataPath, repetition)

    df['Condition'] = conditions
    df['Value'] = values
    df['Distance'] = distances
   
#%%

fullpath = ImageAnalysisCode.GetFullFilePaths(dataPath)

if camera == 'Andor':
    metaData = ImageAnalysisCode.ExtractMetaData(fullpath)
else:
    metaData = None
   
images = ImageAnalysisCode.GetImages(fullpath, camera, ROI, metaData)
images_corrected = LSAnalysisCode.BGsubtraction_alt(images, 10)


# empty lists to store fitted parameters
Xcenters = []; Ycenters = []; Xwidths = []; Ywidths = []; Xamps = []; Yamps = []

for image_arr in images:
   
    image_arr, _ = ImageAnalysisCode.Rotate(image_arr, angle)
    paramX, paramY = ImageAnalysisCode.FitGaussian(image_arr, doPlot, 'Wide')
   
    Xcenter = paramX[0]*pixSize
    Xwidth = paramX[1]*pixSize
   
    Ycenter = paramY[0]*pixSize
    Ywidth = paramY[1]*pixSize
   
    Xcenters.append(Xcenter); Ycenters.append(Ycenter)
    Xwidths.append(Xwidth); Ywidths.append(Ywidth)
    Xamps.append(paramX[2]); Yamps.append(paramY[2])
       
df['Xcenter'] = Xcenters; df['Ycenter'] = Ycenters
df['Xwidth'] = Xwidths; df['Ywidth'] = Ywidths
df['Xamp'] = Xamps; df['Yamp'] = Yamps    

#%%

colsForAnalysis = ['Xwidth', 'Ywidth']

if df['Value'].isna().any():
    stats = df.groupby(['Distance'])[colsForAnalysis].agg(['mean', 'std']).reset_index()
    stats.columns = ['Distance'] + ['_'.join(col).strip() for col in stats.columns[1:]]
else:
    stats = df.groupby(['Distance', 'Value'])[colsForAnalysis].agg(['mean', 'std']).reset_index()
    stats.columns = ['Distance', 'Value'] + ['_'.join(col).strip() for col in stats.columns[2:]]

# convert distance and waists to meters
stats['Distance'] = stats['Distance']*1e-3
width_cols = [col for col in stats.columns if 'width' in col]
stats[width_cols] = stats[width_cols] * 1e-6


#%%

def analyze_all_powers(stats, colsForAnalysis=['Xwidth', 'Ywidth'], wavelength=1064e-9):
    results = []

    def w_z(z, w0, z0):
        # Rayleigh range formula
        zR = np.pi * w0**2 / wavelength
        return w0 * np.sqrt(1 + ((z - z0) / zR)**2)

    # Group by 'Value' (Power)
    for power, group in stats.groupby('Value'):
        row = {'Power': power}
        
        for col in colsForAnalysis:
            z = group['Distance'].values
            w_meas = group[f'{col}_mean'].values
            w_err = group[f'{col}_std'].values
            
            # Initial guess: [min width, position of min width]
            p0 = [np.min(w_meas), z[np.argmin(w_meas)]]
            
            try:
                popt, pcov = curve_fit(w_z, z, w_meas, p0=p0, sigma=w_err, absolute_sigma=True)
                perr = np.sqrt(np.diag(pcov))
                
                # Store results in the dictionary
                row[f'{col}_w0'] = popt[0]
                row[f'{col}_z0'] = popt[1]
                row[f'{col}_w0_err'] = perr[0]
                row[f'{col}_z0_err'] = perr[1]
                
            except Exception as e:
                print(f"Fit failed for Power {power}, Column {col}: {e}")
                row[f'{col}_w0'] = row[f'{col}_z0'] = np.nan

        results.append(row)

    return pd.DataFrame(results)

def plot_results(fit_results):
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot w0 vs Power
    ax[0].errorbar(fit_results['Power'], fit_results['Xwidth_w0']*1e6, 
                 yerr=fit_results['Xwidth_w0_err']*1e6, fmt='o-', label='X waist', capsize=2)
    ax[0].errorbar(fit_results['Power'], fit_results['Ywidth_w0']*1e6, 
                 yerr=fit_results['Ywidth_w0_err']*1e6, fmt='o-', label='Y waist', capsize=2)
    ax[0].set_xlabel('Power (Value)')
    ax[0].set_ylabel('Waist $w_0$ (μm)')
    ax[0].legend()
    ax[0].set_title('Waist vs Power')

    # Plot z0 vs Power
    ax[1].errorbar(fit_results['Power'], fit_results['Xwidth_z0']*1e3, 
                 yerr=fit_results['Xwidth_z0_err']*1e3, fmt='o-', label='X position', capsize=2)
    ax[1].errorbar(fit_results['Power'], fit_results['Ywidth_z0']*1e3, 
                 yerr=fit_results['Ywidth_z0_err']*1e3, fmt='o-', label='Y position', capsize=2)
    ax[1].set_xlabel('Power (Value)')
    ax[1].set_ylabel('Focus Position $z_0$ (mm)')
    ax[1].legend()
    ax[1].set_title('Focus Position vs Power')
    
    plt.tight_layout()
    plt.show()


# Execute the analysis
fit_results = analyze_all_powers(stats)
plot_results(fit_results)
