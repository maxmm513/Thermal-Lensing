import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ImageAnalysis import ImageAnalysisCode
import os
import re
import matplotlib.lines as mlines
import cv2

def RecognizeCommonPhrase(dataPathList, repetition):
    
    pattern_both = re.compile(r'(?:(\d+(?:\.\d+)?)\s*mm).*?power\s*(\d+(?:\.\d+)?)$', re.IGNORECASE)
    pattern_distance_only = re.compile(r'(\d+(?:\.\d+)?)\s*mm', re.IGNORECASE)

    conditions = []
    values = []
    distances = []

    for name in dataPathList:
        basename = os.path.basename(name)

        match_both = pattern_both.search(basename)
        match_dist = pattern_distance_only.search(basename)

        if match_both:
            distance = float(match_both.group(1))
            value = float(match_both.group(2))
            condition = re.sub(pattern_both, '', basename).strip()
        elif match_dist:
            distance = float(match_dist.group(1))
            value = np.nan
            condition = re.sub(pattern_distance_only, '', basename).strip()
        else:
            distance = np.nan
            value = np.nan
            condition = basename.strip()

        conditions.extend([condition] * repetition)
        values.extend([value] * repetition)
        distances.extend([distance] * repetition)
    
    return conditions, values, distances



def Rotate(image_arr, deg):
    height, width = image_arr.shape[:2]
    
    center = (width / 2, height / 2)
    
    rotationMatrix = cv2.getRotationMatrix2D(center, deg, 1.0)
    rotated = cv2.warpAffine(image_arr, rotationMatrix, (width, height))
    
    return rotated, rotationMatrix



def Gauss1D(x,xc,sigX,A, offset):
    G = A * np.exp(-2 * (x-xc)**2 / sigX**2) + offset
    return G



def FitGaussianImage(gaussImageFile, graph=True, graphOption='Wide'):
    
    beam = ImageAnalysisCode.CheckFile(gaussImageFile)
    Ny, Nx = beam.shape
    x_index = np.linspace(0, Nx-1, Nx)
    y_index = np.linspace(0, Ny-1, Ny)
    
    max_index = np.unravel_index(np.argmax(beam), beam.shape)
    max_x, max_y = max_index

    vert = beam[:, max_y]
    horiz = beam[max_x, :]
    
    sigGuess = 40
    offset = 0
    
    guessX = [max_y, sigGuess, np.max(horiz), offset]
    paramX,_ = curve_fit(Gauss1D, x_index, horiz, p0=guessX)
    paramX[1] = np.abs(paramX[1]) # ensure positive width
    x_fit1 = np.linspace(0, Nx-1, 5000)
    y_fit1 = Gauss1D(x_fit1, paramX[0], paramX[1], paramX[2], paramX[3])

    guessY = [max_x, sigGuess, np.max(vert), offset]
    paramY,_ = curve_fit(Gauss1D, y_index, vert, p0=guessY)
    paramY[1] = np.abs(paramY[1])
    x_fit2 = np.linspace(0,Ny-1, 5000)
    y_fit2 = Gauss1D(x_fit2, paramY[0], paramY[1], paramY[2], paramY[3])
    
    centerX = int(paramX[0])
    centerY = int(paramY[0])
            
    if graph:
        fig, ax = plt.subplots(1,3)
        
        ax[1].plot(x_fit1, y_fit1,'r',linewidth=3)
        ax[1].scatter(x_index, horiz, s=20)
        ax[1].set_title('Fit vs. X')
        
        text_x = f"x0 = {int(paramX[0])} \nσ = {paramX[1]:.2f} px \nA = {paramX[2]:.2f}"
        ax[1].text(0.35, 0.95, text_x, transform=ax[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        ax[2].plot(x_fit2,y_fit2,'r',linewidth=3)
        ax[2].scatter(y_index, vert, s=20)
        ax[2].set_title('Fit vs. Y')
        
        text_y = f"y0 = {int(paramY[0])} \nσ = {paramY[1]:.2f} px \nA = {paramY[2]:.2f}"
        ax[2].text(0.05, 0.95, text_y, transform=ax[2].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        if graphOption == 'Narrow':
            
            ax[1].set_xlim(paramX[0]-4*paramX[1], paramX[0]+4*paramX[1])
            ax[2].set_xlim(paramY[0]-4*paramY[1], paramY[0]+4*paramY[1])
            
            ax[0].imshow(beam, extent=[paramX[0]-1, 
                                       paramX[0]+1, 
                                       paramY[0]-1, 
                                       paramY[0]+1])
        else:
            ax[0].imshow(beam*-1,cmap='binary')
            ax[0].imshow(beam,cmap='jet')
        
        ax[0].set_title('Image')        
    return paramX, paramY



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

        conditions, values, distances = RecognizeCommonPhrase(dataPath, repetition)

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
       
        image_arr, _ = Rotate(image_arr, angle)
        paramX, paramY = FitGaussianImage(image_arr, doPlot, 'Wide')
       
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
            if power > 11:
                fig.suptitle(f'P={power}%', fontsize=16, weight='bold')
            else:
                fig.suptitle(f'P={power} V', fontsize=16, weight='bold')
            j = 0

        for col in colsForAnalysis:
            z = group['Distance'].values
            w_meas = group[f'{col}_mean'].values
            w_err = group[f'{col}_std'].values
            
            # if a fit gives an error of zero, replace it with a small value
            w_err[w_err == 0] = 1e-8

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



def Fit_GaussianBeamRadius_M2factor(stats, colsForAnalysis, wavelength=1064e-9, doPlot=False):
    '''
    Modified version of Fit_GaussianBeamRadius but includes M^2 beam quality factor
    in the definition for the Rayleigh range. This is an additional fit parameter that
    will be accounted for in the output dataframe
    '''
    results = []
    
    # Gaussian beam radius function w(z)
    def w_z(z, w0, z0, M2):
        zR = np.pi * w0**2 / (M2*wavelength)
        return w0 * np.sqrt(1 + ((z - z0) / zR)**2)

    # group by Power
    for power, group in stats.groupby('Power'):
        row = {'Power': power}
        
        if doPlot:
            fig,ax = plt.subplots(1,2,figsize=(8,4))
            if power > 11:
                fig.suptitle(f'P={power}%', fontsize=16, weight='bold')
            else:
                fig.suptitle(f'P={power} V', fontsize=16, weight='bold')
            j = 0

        for col in colsForAnalysis:
            z = group['Distance'].values
            w_meas = group[f'{col}_mean'].values
            w_err = group[f'{col}_std'].values
            
            # if a fit gives an error of zero, replace it with a small value
            w_err[w_err == 0] = 1e-8

            # [min width, position of min width, M2 guess]
            p0 = [np.min(w_meas), z[np.argmin(w_meas)], 1]
            
            try:
                popt, pcov = curve_fit(w_z, z, w_meas, p0=p0, sigma=w_err, absolute_sigma=True)
                perr = np.sqrt(np.diag(pcov))
                
                # store results in the dictionary
                axis = col[0]
                row[f'w0_{axis} fit'] = popt[0]
                row[f'w0_{axis} fit err'] = perr[0]
                row[f'z0_{axis} fit'] = popt[1]
                row[f'z0_{axis} fit err'] = perr[1]
                row[f'M2_{axis}'] = popt[2]
                row[f'M2_{axis} err'] = perr[2]
                
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
                txt = f'w0={popt[0]*1e6:.2f} $\pm$ {perr[0]*1e6:.2f} μm\nz0={popt[1]*1e3:.2f} $\pm$ {perr[1]*1e3:.2f} mm\n$M^2$={popt[2]:.3f} $\pm$ {perr[2]:.3f}'
                ax[j].text(0.25, 0.8, txt, transform=ax[j].transAxes, fontsize=10, bbox=dict(facecolor='white'))
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

#%%

def Plot_BeamEvolutionXYWithImages(group, folders, power, camera, ROI, crop_window=150, wavelength=1064e-9):
    '''
    Plots X and Y beam radius vs position as stacked subplots, and includes an 
    auto-cropped inset of the beam image from each position aligned below.
    '''
    
    # Gaussian beam radius function w(z)
    def w_z(z, w0, z0):
        zR = np.pi * w0**2 / wavelength
        return w0 * np.sqrt(1 + ((z - z0) / zR)**2)

    z = group['Distance'].values
    z_fit = np.linspace(min(z), max(z), 2000)
    z_mm_arr = z * 1e3

    # 2 rows, 1 column
    fig, (ax_X, ax_Y) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.subplots_adjust(bottom=0.25, hspace=0.05) # leave room at bottom, remove gap between plots
    fig.suptitle(f'P = {power}%', fontsize=16, weight='bold')

    # --- Fit and Plot Xwidth ---
    w_meas_X = group['Xwidth_mean'].values
    w_err_X = group['Xwidth_std'].values
    p0_X = [np.min(w_meas_X), z[np.argmin(w_meas_X)]]
    
    try:
        popt_X, pcov_X = curve_fit(w_z, z, w_meas_X, p0=p0_X, sigma=w_err_X, absolute_sigma=True)
        perr_X = np.sqrt(np.diag(pcov_X))
        
        ax_X.errorbar(z*1e3, w_meas_X*1e6, yerr=w_err_X*1e6, fmt='o', capsize=3)
        ax_X.plot(z_fit*1e3, w_z(z_fit, *popt_X)*1e6, 'r-')
        txt_X = f'w0x = {popt_X[0]*1e6:.2f} $\\pm$ {perr_X[0]*1e6:.2f} μm\nz0x = {popt_X[1]*1e3:.2f} $\\pm$ {perr_X[1]*1e3:.2f} mm'
        ax_X.text(0.4, 0.80, txt_X, transform=ax_X.transAxes, bbox=dict(facecolor='white', alpha=0.9))
    except Exception as e:
        print(f"X Fit failed for Power {power}: {e}")

    ax_X.set_ylabel('Xwidth (μm)', fontsize=12)
    ax_X.grid(True, alpha=0.3)

    # --- Fit and Plot Ywidth ---
    w_meas_Y = group['Ywidth_mean'].values
    w_err_Y = group['Ywidth_std'].values
    p0_Y = [np.min(w_meas_Y), z[np.argmin(w_meas_Y)]]
    
    try:
        popt_Y, pcov_Y = curve_fit(w_z, z, w_meas_Y, p0=p0_Y, sigma=w_err_Y, absolute_sigma=True)
        perr_Y = np.sqrt(np.diag(pcov_Y))
        
        ax_Y.errorbar(z*1e3, w_meas_Y*1e6, yerr=w_err_Y*1e6, fmt='o', capsize=3, color='C2')
        ax_Y.plot(z_fit*1e3, w_z(z_fit, *popt_Y)*1e6, 'r-')
        txt_Y = f'w0y = {popt_Y[0]*1e6:.2f} $\\pm$ {perr_Y[0]*1e6:.2f} μm\nz0y = {popt_Y[1]*1e3:.2f} $\\pm$ {perr_Y[1]*1e3:.2f} mm'
        ax_Y.text(0.4, 0.80, txt_Y, transform=ax_Y.transAxes, bbox=dict(facecolor='white', alpha=0.9))
    except Exception as e:
        print(f"Y Fit failed for Power {power}: {e}")

    ax_Y.set_ylabel('Ywidth (μm)', fontsize=12)
    ax_Y.set_xlabel('Distance (mm)', fontsize=12)
    ax_Y.grid(True, alpha=0.3)

    # --- Draw Canvas for image insets ---
    fig.canvas.draw()
    bbox = ax_Y.get_position()
    xlim = ax_Y.get_xlim()

    img_w = 0.08  
    img_h = 0.12  
    y_bottom = 0.04 

    for z_mm, folder in zip(z_mm_arr, folders):
        try:
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            if not files:
                continue
            
            files.sort()
            last_img_path = os.path.join(folder, files[-1])
            
            metaData = None
            if camera == 'Andor':
                metaData = ImageAnalysisCode.ExtractMetaData([last_img_path])

            img_list = ImageAnalysisCode.GetImages([last_img_path], camera, ROI, metaData)
            if not img_list:
                continue
                
            img_arr = img_list[0]
            
            # auto-crop
            cy, cx = np.unravel_index(np.argmax(img_arr), img_arr.shape)
            y0 = max(0, cy - crop_window // 2)
            y1 = min(img_arr.shape[0], cy + crop_window // 2)
            x0 = max(0, cx - crop_window // 2)
            x1 = min(img_arr.shape[1], cx + crop_window // 2)
            img_cropped = img_arr[y0:y1, x0:x1]
            
            # figure mapping relative to the bottom plot
            x_fig = bbox.x0 + bbox.width * (z_mm - xlim[0]) / (xlim[1] - xlim[0])
            
            ax_img = fig.add_axes([x_fig - img_w/2, y_bottom, img_w, img_h])
            ax_img.imshow(img_cropped, cmap='turbo') 
            ax_img.axis('off')
            
            # draw line connecting the horiz axis of ax_Y to the images
            line = mlines.Line2D([x_fig, x_fig], [y_bottom + img_h, bbox.y0 - 0.01], 
                                 color='gray', linestyle=':', lw=1.5, transform=fig.transFigure)
            fig.add_artist(line)
            
        except Exception as e:
            print(f"Could not load raw image from {folder}: {e}")