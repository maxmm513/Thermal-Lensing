import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ImageAnalysis import ImageAnalysisCode


def Fit_GaussianRawImages(dataPath, camera, ROI, 
            repetition=6, 
            commonPhrase=True, 
            angle=0, 
            doPlot=False
            ):
    '''
    Generates dataframe of Gaussian fit results (Xwidth, Xcenter, etc) for all
    raw images in the 'dataPath' list

    Parameters
    ----------
    dataPath : list of folders with images to analyze. Each folder corresponds
                to beam measurements at a single position and power, and contains
                multiple images that will be averaged over in a separate function
    camera : string, name of camera that's taking pictures
    ROI : list of image coordinates to crop the raw array
    repetition : integer, number of images in each folder in dataPath
    commonPhrase : TYPE, optional
    angle : float, angle to rotate the image by
    doPlot : optional bool, displays the raw image and Gaussian fits along each axis

    Returns
    -------
    df : dataframe of length len(dataPath)*repetition, containing results from gaussian
            fits to the beam (widths, centers, etc)
    '''
    
    df = pd.DataFrame(columns=['File', 'Condition', 'Power', 'Xcenter', 'Ycenter', 'Xwidth', 'Ywidth', 'Xamp', 'Yamp'])

    if commonPhrase:

        conditions, values, distances = ImageAnalysisCode.RecognizeCommonPhrase(dataPath, repetition)

        df['Condition'] = conditions
        df['Power'] = values
        df['Distance'] = distances

    fullpath = ImageAnalysisCode.GetFullFilePaths(dataPath)
    
    # pixels sizes
    if camera == 'Basler':
        pixSize = 2 #um/px
    elif camera == 'FLIR':
        pixSize = 3.75 #um/px
    elif camera == 'Andor':
        pixSize = 6.5 #um/pix
        
    # metadata (needed for Andor camera)
    if camera == 'Andor':
        metaData = ImageAnalysisCode.ExtractMetaData(fullpath)
    else:
        metaData = None
       
    images = ImageAnalysisCode.GetImages(fullpath, camera, ROI, metaData)


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
    
    return df


def RawFitStats(df, colsForAnalysis):
    '''
    Returns dataframe of averaged beam widths, centers, etc

    Parameters
    ----------
    df : dataframe of results from Gaussian fitting of raw images
    colsForAnalysis : list of columns to analyze, e.g. ['Xwidth', 'Ywidth']

    Returns
    -------
    stats : dataframe of averaged beam properties, average results from the 
        several images in a single folder
    '''
    
    if df['Power'].isna().any():
        stats = df.groupby(['Distance'])[colsForAnalysis].agg(['mean', 'std']).reset_index()
        stats.columns = ['Distance'] + ['_'.join(col).strip() for col in stats.columns[1:]]
    else:
        stats = df.groupby(['Distance', 'Power'])[colsForAnalysis].agg(['mean', 'std']).reset_index()
        stats.columns = ['Distance', 'Power'] + ['_'.join(col).strip() for col in stats.columns[2:]]

    # convert distance and waists to meters
    stats['Distance'] = stats['Distance']*1e-3
    width_cols = [col for col in stats.columns if 'width' in col]
    stats[width_cols] = stats[width_cols] * 1e-6
    
    return stats


def Fit_GaussianBeamRadius(stats, colsForAnalysis, wavelength=1064e-9, doPlot=False):
    '''
    Returns dataframe of z0, w0 vs. power for input dataframe of 
    beam widths vs. position

    Parameters
    ----------
    stats : dataframe of gaussian fit results from raw beam images (widths, centers, errors etc)
    colsForAnalysis : [Xwidth, Ywidth] the dataframe columns to analyze
    doPlot : optional to plot beam radius vs. distance with w(z) fit

    Returns
    -------
    dataframe of w0, z0 at different powers, and errors of w0, z0 from fit's pcov
    '''
    results = []
    
    # Gaussian beam radius function w(z)
    def w_z(z, w0, z0):
        zR = np.pi * w0**2 / wavelength
        return w0 * np.sqrt(1 + ((z - z0) / zR)**2)

    # group by Power
    for power, group in stats.groupby('Power'):
        row = {'Power': power}
        
        if doPlot:
            fig,ax = plt.subplots(1,2,figsize=(8,4))
            fig.suptitle(f'P={power}%', fontsize=16, weight='bold')
            j = 0

        for col in colsForAnalysis:
            z = group['Distance'].values
            w_meas = group[f'{col}_mean'].values
            w_err = group[f'{col}_std'].values

            # [min width, position of min width]
            p0 = [np.min(w_meas), z[np.argmin(w_meas)]]
            
            try:
                popt, pcov = curve_fit(w_z, z, w_meas, p0=p0, sigma=w_err, absolute_sigma=True)
                perr = np.sqrt(np.diag(pcov))
                
                # store results in the dictionary
                axis = col[0]
                row[f'w0_{axis} fit'] = popt[0]
                row[f'w0_{axis} fit err'] = perr[0]
                row[f'z0_{axis} fit'] = popt[1]
                row[f'z0_{axis} fit err'] = perr[1]
                
            except Exception as e:
                print(f"Fit failed for Power {power}, Column {col}: {e}")
                row[f'w0_{axis}'] = row[f'z0_{axis}'] = np.nan
                
            if doPlot:
                z_fit = np.linspace(min(z), max(z), 2000)
                
                ax[j].errorbar(z*1e3, w_meas*1e6, yerr=w_err*1e6, fmt='o', capsize=3)
                ax[j].plot(z_fit*1e3, w_z(z_fit, *popt)*1e6,'r-')
                
                ax[j].set_title(f'{col}',fontsize=14)
                ax[j].set_xlabel('Distance (mm)')
                ax[j].set_ylabel(col+' (μm)')
                ax[j].grid(True,alpha=0.2)
                txt = f'w0={popt[0]*1e6:.2f} $\pm$ {perr[0]*1e6:.2f} μm\nz0={popt[1]*1e3:.2f} $\pm$ {perr[1]*1e3:.2f} mm'
                ax[j].text(0.25, 0.85, txt, transform=ax[j].transAxes, bbox=dict(facecolor='white'))
                plt.tight_layout()
                j=+1
                

        results.append(row)

    return pd.DataFrame(results)



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
        
    plt.figure(figsize=(4.5, 3.5))    

    col_vals = quant+' fit'
    col_errs = quant+' fit err'
    
    plt.errorbar(P_W, results[col_vals]*scale, yerr=results[col_errs]*scale, fmt='o', capsize=3)
    
    plt.title(quant+' shift vs power', fontsize=14)
    plt.xlabel('Power (W)')
    plt.ylabel(quant + f' ({unit})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

