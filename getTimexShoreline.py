
# Shoreline Extraction Image Product | [getTimexShoreline.py]

### getStationInfo(ssPath) | return(stationInfo) - [imgProcTools.py]
import cv2
import pytz
import termcolor
from tqdm import trange
from pathlib import Path
from termcolor import cprint

def getStationInfo(ssPath):
    # Converts specific fields in the loaded stationInfo dictionary from lists to NumPy arrays.
    setupFile = open(ssPath)
    stationInfo = json.load(setupFile)
    
    stationInfo['Apx. Shoreline']['slX'] = np.asarray(stationInfo['Apx. Shoreline']['slX'])
    stationInfo['Apx. Shoreline']['slY'] = np.asarray(stationInfo['Apx. Shoreline']['slY'])
    
    stationInfo['Collision Test Points']['x'] = np.asarray(stationInfo['Collision Test Points']['x'])
    stationInfo['Collision Test Points']['y'] = np.asarray(stationInfo['Collision Test Points']['y'])
    
    stationInfo['Dune Line Info']['Dune Line Interpolation'] = np.asarray(stationInfo['Dune Line Info']['Dune Line Interpolation'])
    stationInfo['Dune Line Info']['Dune Line Points'] = np.asarray(stationInfo['Dune Line Info']['Dune Line Points'])
    
    stationInfo['Shoreline Transects']['x'] = np.asarray(stationInfo['Shoreline Transects']['x'])
    stationInfo['Shoreline Transects']['y'] = np.asarray(stationInfo['Shoreline Transects']['y'])
    
    return(stationInfo)

### mapROI(stationInfo, photo, imgType): | return (maskedImg, figROI) - [shorelineFunctions.py]
import itertools
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.measure import profile_line
from statsmodels.nonparametric.kde import KDEUnivariate

def mapROI(stationInfo, photo, imgType):
    # Set width and height in pixels of image.
    w = len(photo[1])
    h = len(photo)
    
    # Retrieves pixel transect coordinates based on image type.
    if imgType == 'avg' or imgType == 'brt' or imgType == 'snap':
        transects = stationInfo['Shoreline Transects']
        xt = np.asarray(transects['x'])
        yt = np.asarray(transects['y'])       
    elif imgType == 'rec':
        transects = stationInfo['Rectified Transects']
        xt = np.asarray(transects['x'])
        yt = np.asarray(transects['y'])
    
    # Draws polygon based on transects and creates masks on unneeded areas.
    cords = []
    for i in range(0,len(xt)):
        pts = [int(xt[i,1]),int(yt[i,1])]
        cords.append(pts)
        
    for i in range(len(xt)-1,-1,-1):
        pts = [int(xt[i,0]),int(yt[i,0])]
        cords.append(pts)
    cords.append(cords[0])
    
    xs, ys = zip(*cords)
    poly = list(itertools.chain.from_iterable(cords))
    
    img = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    
    # Converts mask to a nparray.
    if imgType == 'rec':
        mask = np.array(np.flipud(img))
    else:
        mask = np.array(img)
    
    # Sets pixels outside of mask to NaN.
    maskedImg = photo.astype(np.float64)
    for i in range(0,w):
        for j in range(0,h):
            if mask[j,i] == 0:
                maskedImg[j,i] = np.NaN
            else:
                maskedImg[j,i] = (maskedImg[j,i]/255)
    
    # Generates figure with mask.
    figROI = pltFig_ROI(stationInfo, photo, maskedImg, xs, ys, imgType)
    
    return (maskedImg, figROI)

### pltFig_ROI(stationInfo, photo, maskedImg, xs, ys, imgType): | return(figROI) - [plotFigures.py]
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable

def pltFig_ROI(stationInfo, photo, maskedImg, xs, ys, imgType):
        
    # Initializes station setup.
    stationname = stationInfo['Station Name']
    dtInfo = stationInfo['Datetime Info']
    date = str(dtInfo.date())
    time = str(dtInfo.hour) + str(dtInfo.minute)
    
    if imgType == 'avg':
        source = 'Time-Averaged'
    if imgType == 'brt': 
        source = 'Brightest Pixel'
    if imgType == 'rec':
        source = 'Rectified Image'
    if imgType == 'snap':
        source = 'Snapshot'    
    if imgType == 'avg' or imgType == 'brt' or imgType == 'snap':
        rmb = maskedImg[:,:,0] - maskedImg[:,:,2]
        
        # Plots figure for final output.
        plt.ioff()
        figROI, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,9))
        ax1.imshow(photo)
        ax1.plot(xs,ys, 'r--', label = 'ROI Boundaries') 
        ax1.set_title(('Region of Interest (ROI)'), fontsize = 14)
        ax1.set_xlabel('Image Width (pixels)')
        ax1.set_ylabel('Image Height (pixels)')
        ax1.legend(prop={'size': 9}, loc=2)
        ax1.tick_params(axis = 'both', which = 'major', labelsize = 10)
        ax1.tick_params(axis = 'both', which = 'minor', labelsize = 10)
        im2 = ax2.imshow(rmb, cmap=plt.get_cmap('viridis'))
        ax2.plot(xs,ys, 'r--', label = 'ROI Boundaries') 
        ax2.set_title(('Red - Blue ROI Masking'), fontsize = 14)
        ax2.set_xlabel('Image Width (pixels)')
        ax2.set_ylabel('Image Height (pixels)')
        ax2.legend(prop={'size': 9}, loc=2)
        ax2_divider = make_axes_locatable(ax2)
        ax2.tick_params(axis = 'both', which = 'major', labelsize = 10)
        ax2.tick_params(axis = 'both', which = 'minor', labelsize = 10)
        cax2 = ax2_divider.append_axes("right", size="5%", pad="2%")
        figROI.colorbar(im2, cax=cax2)
        figROI.suptitle((stationname.capitalize() + ' Region of Interest\n' 
                         ' at ' + time[:2] + ':' + time[2:] + ' UTC' + ' on ' + date + 
                         ' (' + source + ')'), fontsize = 16, y = 0.77)
        plt.tight_layout()
    plt.close()
    
    return(figROI)

### improfile(rmb, stationInfo): | return(P) - [shorelineFunctions.py]
def improfile(rmb, stationInfo):
    
    # Extract intensity profiles along shoreline transects from an input image.
    transects = stationInfo['Shoreline Transects']
    xt = np.asarray(transects['x'])
    yt = np.asarray(transects['y'])
  
    n = len(xt)
    imProf = [0]*n   
    for i in range(0,n):
        imProf[i] = profile_line(rmb, (yt[i,1], xt[i,1]), (yt[i,0], xt[i,0]),
                                 mode = 'constant')
        
    tot = [imProf[0].tolist()]
    
    for i in range(0,n): 
        cvt = imProf[i].tolist()
        tot.append(cvt)

    improfile = []
    for i in range(0,n): 
        for j in range(0, len(tot[i])):
            if math.isnan(tot[i][j]):
                pass
            else:
                improfile.append(tot[i][j])   
            
    P = np.asarray(improfile)  
    
    return(P)

### ksdensity(P, **kwargs): | return (pdf, x_grid) - [shorelineFunctions.py]
def ksdensity(P, **kwargs):
    # Univariate Kernel Density Estimation with Statsmodels.
    x_grid = np.linspace(P.max(), P.min(), 1000)
    kde = KDEUnivariate(P)
    kde.fit(**kwargs)
    pdf = kde.evaluate(x_grid)
    
    return (pdf, x_grid)

### extract() | return(shoreline) - [slExtract.py]
import os
import json
import numpy as np
from datetime import datetime, timedelta

def extract(stationInfo, rmb, maskedImg, threshInfo, imgType):
    
    # Defines variables.
    stationname = stationInfo['Station Name']
    slTransects = stationInfo['Shoreline Transects']
    dtInfo = stationInfo['Datetime Info']
    date = dtInfo.date()
    xt = np.asarray(slTransects['x'])
    yt = np.asarray(slTransects['y'])
    orn = stationInfo['Orientation']
    thresh = threshInfo['Thresh'] 
    thresh_otsu = threshInfo['Otsu Threshold']
    thresh_weightings = threshInfo['Threshold Weightings']
    length = len(yt) #- 1
    trsct = range(0, length)

    # Initializes list to store values.
    values = [0]*length
    revValues = [0]*length
    yList = [0]*length
    xList = [0]*length
    
    # Checks orientation.
    if orn == 0:
        # Finds the index of the first occurrence of
        # a value greater than thresh_otsu in a list.
        def find_intersect(List, thresh):
            res = [k for k, x in enumerate(List) if x > thresh_otsu]
            return None if res == [] else res[0]
    
        # Loops over each transect and extracts values from rmb 
        # array based on transect coordinates. Calculates the 
        # difference between yMax and yMin to determine the 
        # number of elements in the val list. Then populates the val list.
        for i in trsct:
            x = int(xt[i][0])
            yMax = round(yt[i][1])
            yMin = round(yt[i][0])
            y = yMax-yMin
            yList[i] = np.zeros(shape=y)
            val = [0]*(yMax-yMin)
            for j in range(0,len(val)):
                k = yMin + j
                val[j] = rmb[k][x]
            val = np.array(val)
            values[i] = val
        
        # Finding the index of the intersection point with the threshold value. 
        # It then calculates the x and y coordinates of the intersection point and stores.
        intersect = [0]*len(xt)  
        idx = [0]*len(xt) 
        Pts = [0]*len(xt) 
        xPt = [0]*len(xt)
        yPt = [0]*len(xt)  
        
        for i in range(0, len(values)):
            intersect[i] = find_intersect(values[i], thresh_otsu)
            idx[i] = np.where(values[i][intersect[i]] == values[i])
            n = len(idx[i][0])-1
            if len(idx[i][0]) == 0:
                yPt[i] = None
                xPt[i] = None
            else:
               yPt[i] = min(yt[i]) + idx[i][0][n]
               xPt[i] = int(xt[i][0])
               Pts[i] = (xPt[i], yPt[i]) 
            
            # Calculates the average value in a 21x21 sample area around each 
            # intersection point.
            areaAvg = [0]*len(Pts)
            sample = np.zeros((21,21))
            
            for i in range(0,len(Pts)):
                if Pts[i] == 0:
                    pass
                else:
                    orginX = int(Pts[i][0])
                    orginY = int(Pts[i][1])
                    xBounds = range(orginX - 10, orginX + 11)
                    yBounds = range(orginY - 10, orginY + 11)
                    for j in yBounds:
                        a = j - orginY
                        for k in xBounds:
                            b = k - orginX
                            sample[a][b] = rmb[j][k]
                    areaAvg[i] = np.mean(sample)
            
            # Determines the points that are considered as valid shoreline 
            # points based on the average values calculated earlier. 
            # It removes points that fall outside a certain range around the threshold value.
            # Exports shoreline variables to a JSON file. Returns the shoreline points array.
            buffer = (float(thresh_otsu) * .20)
            exc = {0}
            for i in range(0,len(Pts)):
                if abs((buffer + thresh_otsu)) > abs(areaAvg[i]) > abs((buffer - thresh_otsu)):
                    pass
                else:
                    exc.add(i)
                                            
            truePts = [v for i, v in enumerate(Pts) if i not in exc]  
                
            threshX = [0]*len(truePts)
            threshY = [0]*len(truePts)
            for i in range(0, len(truePts)):
                threshX[i] = truePts[i][0]
                threshY[i] = truePts[i][1]
             
            threshX = np.array(threshX)
            threshY = np.array(threshY)
            
            shoreline = np.vstack((threshX,threshY)).T

    else:
        # Finds the index of the first element in List less than thresh.
        def find_intersect(List, thresh):
            res = [k for k, x in enumerate(List) if x < thresh_otsu]
            return None if res == [] else res[0]
        
        # Iterates over each transect in the list and creates array of points.
        for i in trsct:
            xMax = round(xt[i][0])
            y = int(yt[i][0])
            yList[i] = np.full(shape=xMax, fill_value= y)
            xList[i] = np.arange(xMax)
            values[i] =rmb[y][0:xMax]
            revValues[i] = rmb[y][::-1]
        
        # Creates empty arrays.
        intersect = [0]*len(yt)  
        idx = [0]*len(yt) 
        Pts = [0]*len(yt) 
        xPt = [0]*len(yt)
        yPt = [0]*len(yt)  
        
        # Checks revValues againts thresh_otsu.
        for i in range(0, len(values)):
            intersect[i] = find_intersect(revValues[i], thresh_otsu)
            idx[i] = np.where(revValues[i][intersect[i]] == values[i])
            n = len(idx[i][0])-1
            if len(idx[i][0]) == 0:
                xPt[i] = None
                yPt[i] = None
            else:
               xPt[i] = idx[i][0][n]
               yPt[i] = int(yt[i][0])
               Pts[i] = (xPt[i], yPt[i]) 
            
            # Computes a 21x21 pixel average around each point.
            areaAvg = [0]*len(Pts)
            sample = np.zeros((21,21))
            
            # Filters out points in buffer zone.
            for i in range(0,len(Pts)):
                if Pts[i] == 0:
                    pass
                else:
                    orginX = Pts[i][0]
                    orginY = Pts[i][1]
                    xBounds = range(orginX - 10, orginX + 11)
                    yBounds = range(orginY - 10, orginY + 11)
                    for j in yBounds:
                        a = j - orginY
                        for k in xBounds:
                            b = k - orginX
                            sample[a][b] = rmb[j][k]
                    areaAvg[i] = np.mean(sample)

            buffer = (float(thresh_otsu) * .20)
            exc = {0}
            for i in range(0,len(Pts)):
                if abs((buffer + thresh_otsu)) > abs(areaAvg[i]) > abs((buffer - thresh_otsu)):
                    pass
                else:
                    exc.add(i)
            
            # Stores final shoreline values in an array. 
            truePts = [v for i, v in enumerate(Pts) if i not in exc]  
            threshX = [0]*len(truePts)
            threshY = [0]*len(truePts)
            for i in range(0, len(truePts)):
                threshX[i] = truePts[i][0]
                threshY[i] = truePts[i][1]
            threshX = np.array(threshX)
            threshY = np.array(threshY)
            shoreline = np.vstack((threshX,threshY)).T
            
    # Contructs dictionary of shoreline variables.
    slVars = {
        'Station Name':stationname, 
        'Date':str(date), 
        'Time Info':str(dtInfo), 
        'Thresh':thresh, 
        'Otsu Threshold':thresh_otsu,
        'Shoreline Transects': slTransects,
        'Threshold Weightings':thresh_weightings,
        'Shoreline Points':shoreline
        }
    
    # Converts numpy arrays to lists and removes non-serializable objects for JSON.
    # datetime and ndarray
    try:
        del slVars['Time Info']['DateTime Object (UTC)']
        del slVars['Time Info']['DateTime Object (LT)']
    except:
        pass
    
    if type(slVars['Shoreline Transects']['x']) == np.ndarray:
        slVars['Shoreline Transects']['x'] = slVars['Shoreline Transects']['x'].tolist()
        slVars['Shoreline Transects']['y'] = slVars['Shoreline Transects']['y'].tolist()
    else:
        pass
    
    slVars['Shoreline Points'] = slVars['Shoreline Points'].tolist()
    
    # Export shoreline variables to JSON.
    fname = (stationname + '.' + datetime.strftime(dtInfo,'%Y-%m-%d_%H%M') + '.' + imgType + '.slVars.json')
    with open(fname, "w") as f:
        json.dump(slVars, f)
            
    return(shoreline)

### pltFig_tranSL(stationInfo, photo, tranSL, imgType): | return(fig_tranSL) - [plotFigures.py]
def pltFig_tranSL(stationInfo, photo, tranSL, imgType):
    
    # Initializes station setup.
    stationname = stationInfo['Station Name']
    dtInfo = stationInfo['Datetime Info']
    date = str(dtInfo.date())
    time = str(dtInfo.hour) + str(dtInfo.minute)
    
    # Uncomment to plot test points.
    # tst = stationInfo['Collision Test Points']
    # tstX = tst['x']
    # tstY = tst['y']
    
    # Sets up dune line for plotting.
    Di = stationInfo['Dune Line Info']
    orn = stationInfo['Orientation']
    if orn == 0:
        duneInt = Di['Dune Line Interpolation']
        xi = duneInt[:,0]
        py = duneInt[:,1]    
    else:
        duneInt = Di['Dune Line Interpolation']
        xi = duneInt[:,0]
        py = duneInt[:,1]
    
    # Plots shoreline.
    plt.ioff()
    fig_tranSL = plt.figure()
    plt.imshow(photo, interpolation = 'nearest')
    plt.xlabel("Image Width (pixels)", fontsize = 10)
    plt.ylabel("Image Height (pixels)", fontsize = 10)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 8)
    plt.tick_params(axis = 'both', which = 'minor', labelsize = 8)
    plt.plot(tranSL[:,0], tranSL[:,1], color = 'r', linewidth = 2, 
         label = 'Detected Shoreline')
    plt.plot(xi, py, color = 'blue', linewidth = 2, label = 'Baseline', 
         zorder = 4)  
 
    # Uncomment to plot test points.
     # plt.scatter(tstX[:-2], tstY[:-2], 3, color = 'lime', linewidths = 0.5,
               # label = 'Test Points', edgecolors ='k', zorder = 10 )
    
    if imgType == 'avg':
        source = 'Time Averaged'
    if imgType == 'brt': 
        source = 'Brightest Pixel'
    if imgType == 'snap':
        source = 'Snapshot'

    plt.title(('Transect Based Shoreline Detection (' + source + ')\n' + stationname.capitalize() + 
               ' on ' + date + ' at ' + time[:2] + ':' + 
               time[2:] + ' UTC'), fontsize = 12)
    plt.legend(prop={'size': 9})
    plt.tight_layout()
    
    # Save plot.
    saveName = (stationname + '.' + date + '_' + time + '.' + 'tranSL-'
                + imgType + '.fix.png')
    plt.savefig(saveName, bbox_inches = 'tight', dpi=400)
    plt.close()
    
    return(fig_tranSL)

#########################################################################################################

### getTimexShoreline.py
import re
import cv2
import scipy.signal as signal
from skimage.filters import threshold_otsu

def getTimexShoreline(stationName, imgName):
    # Import station config.
    # [imgProcTools.py] getStationInfo(ssPath)
    cwd = os.getcwd()
    stationPath = os.path.join(cwd, stationName + '.config.json')
    stationInfo = getStationInfo(stationPath) 
    
    # Extract date/time information from input video.
    dtObj = datetime.strptime(re.sub('\D', '', imgName), '%Y%m%d%H%M%S')
    stationInfo['Datetime Info'] = dtObj
    
    # Converts time avg image's color scale.
    photoAvg = cv2.cvtColor(cv2.imread(imgName), cv2.COLOR_BGR2RGB)
    
    # Creating an array version of image dimensions for plotting.
    h = len(photoAvg)
    w = len(photoAvg[1])
    xgrid = np.round(np.linspace(0, w, w))
    ygrid = np.round(np.linspace(0, h, h))
    X, Y = np.meshgrid(xgrid, ygrid, indexing = 'xy')
    
    # Maps regions of interest on plot.
    # [shorelineFunctions.py] mapROI()
    maskedImg, figROI = mapROI(stationInfo, photoAvg, 'avg')
    
    # Computes R - B.
    # [shorelineFunctions.py] improfile()
    rmb = maskedImg[:,:,0] - maskedImg[:,:,2]
    P = improfile(rmb, stationInfo)
    P = P.reshape(-1,1)
    
    # Water/sand seperation by computing probability density function.
    # [shorelineFunctions.py] ksdensity(P)
    pdfVals, pdfLocs = ksdensity(P)
    thresh_weightings = [(1/3), (2/3)]
    peaks = signal.find_peaks(pdfVals)
    peakVals = np.asarray(pdfVals[peaks[0]])
    peakLocs = np.asarray(pdfLocs[peaks[0]])  
    
    thresh_otsu = threshold_otsu(P)
    I1 = np.asarray(np.where(peakLocs < thresh_otsu))
    J1, = np.where(peakVals[:] == np.max(peakVals[I1]))
    I2 = np.asarray(np.where(peakLocs > thresh_otsu))
    J2, = np.where(peakVals[:] == np.max(peakVals[I2]))
    
    thresh = (thresh_weightings[0]*peakLocs[J1] +
              thresh_weightings[0]*peakLocs[J2])
    
    thresh = float(thresh)
    
    threshInfo = {
        'Thresh':thresh, 
        'Otsu Threshold':thresh_otsu,
        'Threshold Weightings':thresh_weightings
        }
    
    # Extracts final shoreline.
    # [slExtract.py] extract()
    # [plotFigures.py] pltFig_tranSL()
    tranSL = extract(stationInfo, rmb, maskedImg, threshInfo, 'avg')
    fig_tranSL = pltFig_tranSL(stationInfo, photoAvg, tranSL, 'avg')

    return(tranSL, fig_tranSL)


#########################################################################################################

# USER INPUT:

stationName = 'oakisland_west' #
imgName = 'oakisland_west-2023-04-03-122026Z-timex.png' #

# Call the main().
tranSL, fig_tranSL = getTimexShoreline(stationName, imgName)
