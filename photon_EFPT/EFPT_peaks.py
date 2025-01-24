# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:11:15 2023

@author: Aileen

TO DO: break into a couple files for specific functions, then import those in
"""
import glob
import os
import numpy as np
import pandas as pd
from math import erfc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
# from PHUreader import readPHU
from datetime import datetime

import time
import sys
import struct

plt.rcParams.update({'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

''' returns a PDF of a histogram '''
def countsToPDF(counts,binwidth):
    PDF = (counts/np.sum(counts))
    return PDF

''' calculate nth moment. x is independent variable, y is PDF '''
def nthMoment(x, PDF, n):
    return np.sum(PDF*x**n)

''' trendline fit function '''
def trendline(xdata,ydata):
    x = np.linspace(np.min(xdata),np.max(xdata))
    z = np.polyfit(xdata,ydata,1)
    p = np.poly1d(z)
    return x,p(x),z

''' gumbel distribution with x values, prefactor A, peak location mu, 
    scaling factor beta, and noise k '''
def gumbel(x,A,mu,beta,k):
    # norm = np.sum( (1/beta * np.exp(-( (x-mu)/beta + np.exp(-(x-mu)/beta) ))))
    return A/beta * np.exp(-( (x-mu)/beta + np.exp(-(x-mu)/beta))) + k

''' gumbel sum distribution with x values, prefactor A1 A2, peak location mu1 mu2, 
    scaling factor b1 b2, and noise k '''
def gumbelSum(x,A1,A2,mu1,mu2,b1,b2,k):
    return gumbel(x,A1,mu1,b1,k) + gumbel(x, A2, mu2, b2, 0)

''' linear equation '''    
def y(x,A,B):
    return A*x + B

'''
calculate how noisy the data is based on the difference between the peak height 
and data 100 spaces away
'''
def noiseCalculator(maxLoc, histogram): 
    floor = np.mean(histogram[:maxLoc[0]])
    if floor > 0:
        ratio = histogram[maxLoc][0]/floor
    else:
        ratio = 1e9
        print(floor, histogram[maxLoc][0])
    return 1/ratio
       


"""
# Read_PHU.py    Read PicoQuant Unified TTTR Files
# This is demo code. Use at your own risk. No warranties.
# Keno Goertz, PicoQUant GmbH, February 2018

# Note that marker events have a lower time resolution and may therefore appear 
# in the file slightly out of order with respect to regular (photon) event records.
# This is by design. Markers are designed only for relatively coarse 
# synchronization requirements such as image scanning. 

# T Mode data are written to an output file [filename]
# We do not keep it in memory because of the huge amout of memory
# this would take in case of large files. Of course you can change this, 
# e.g. if your files are not too big. 
# Otherwise it is best process the data on the fly and keep only the results.
"""
def readPHU(inputFileName, outputFileName):
    # Tag Types
    tyEmpty8      = struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
    tyBool8       = struct.unpack(">i", bytes.fromhex("00000008"))[0]
    tyInt8        = struct.unpack(">i", bytes.fromhex("10000008"))[0]
    tyBitSet64    = struct.unpack(">i", bytes.fromhex("11000008"))[0]
    tyColor8      = struct.unpack(">i", bytes.fromhex("12000008"))[0]
    tyFloat8      = struct.unpack(">i", bytes.fromhex("20000008"))[0]
    tyTDateTime   = struct.unpack(">i", bytes.fromhex("21000008"))[0]
    tyFloat8Array = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
    tyAnsiString  = struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
    tyWideString  = struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
    tyBinaryBlob  = struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]
    
    # if len(sys.argv) != 3:
    #     print("USAGE: Read_PHU.py inputfile.PHU outputfile.txt")
    #     sys.exit(0)
    
    inputfile = open(inputFileName, "rb")
    outputfile = open(outputFileName, "w+")
    
    # Check if inputfile is a valid PHU file
    # Python strings don't have terminating NULL characters, so they're stripped
    magic = inputfile.read(8).decode("ascii").strip('\0')
    if magic != "PQHISTO":
        print("ERROR: Magic invalid, this is not a PHU file.")
        sys.exit(0)
    
    version = inputfile.read(8).decode("ascii").strip('\0')
    outputfile.write("Tag version: %s\n" % version)
    
    # Write the header data to outputfile and also save it in memory.
    # There's no do ... while in Python, so an if statement inside the while loop
    # breaks out of it
    tagDataList = []    # Contains tuples of (tagName, tagValue)
    while True:
        tagIdent = inputfile.read(32).decode("ascii").strip('\0')
        tagIdx = struct.unpack("<i", inputfile.read(4))[0]
        tagTyp = struct.unpack("<i", inputfile.read(4))[0]
        if tagIdx > -1:
            evalName = tagIdent + '(' + str(tagIdx) + ')'
        else:
            evalName = tagIdent
        outputfile.write("\n%-40s" % evalName)
        if tagTyp == tyEmpty8:
            inputfile.read(8)
            outputfile.write("<empty Tag>")
            tagDataList.append((evalName, "<empty Tag>"))
        elif tagTyp == tyBool8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if tagInt == 0:
                # outputfile.write("False")
                tagDataList.append((evalName, "False"))
            else:
                # outputfile.write("True")
                tagDataList.append((evalName, "True"))
        elif tagTyp == tyInt8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            outputfile.write("%d" % tagInt)
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyBitSet64:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            outputfile.write("{0:#0{1}x}".format(tagInt,18)) # hex with trailing 0s
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyColor8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            outputfile.write("{0:#0{1}x}".format(tagInt,18)) # hex with trailing 0s
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyFloat8:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            outputfile.write("%-3E" % tagFloat)
            tagDataList.append((evalName, tagFloat))
        elif tagTyp == tyFloat8Array:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            outputfile.write("<Float array with %d entries>" % tagInt/8)
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyTDateTime:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagTime = int((tagFloat - 25569) * 86400)
            tagTime = time.gmtime(tagTime)
            outputfile.write(time.strftime("%a %b %d %H:%M:%S %Y", tagTime))
            tagDataList.append((evalName, tagTime))
        elif tagTyp == tyAnsiString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode("ascii").strip("\0")
            # outputfile.write("%s" % tagString)
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyWideString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode("ascii").strip("\0")
            # outputfile.write("%s" % tagString)
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyBinaryBlob:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            # outputfile.write("<Binary blob with %d bytes>" % tagInt)
            tagDataList.append((evalName, tagInt))
        else:
            print("ERROR: Unknown tag type")
            sys.exit(0)
        if tagIdent == "Header_End":
            break
    
    # Reformat the saved data for easier access
    tagNames = [tagDataList[i][0] for i in range(0, len(tagDataList))]
    tagValues = [tagDataList[i][1] for i in range(0, len(tagDataList))]
    
    # Write histogram data to file
    curveIndices = [tagValues[i] for i in range(0, len(tagNames))\
                    if tagNames[i][0:-3] == "HistResDscr_CurveIndex"]
    for i in curveIndices:
        outputfile.write("\n-----------------------")
        histogramBins = tagValues[tagNames.index("HistResDscr_HistogramBins(%d)" % i)]
        resolution = tagValues[tagNames.index("HistResDscr_MDescResolution(%d)" % i)]
        outputfile.write("\nCurve#  %d" % i)
        outputfile.write("\nnBins:  %d" % histogramBins)
        outputfile.write("\nResol:  %3E" % resolution)
        outputfile.write("\nCounts:")
        for j in range(0, histogramBins):
            try:
                histogramData = struct.unpack("<i", inputfile.read(4))[0]
            except:
                print("The file ended earlier than expected, at bin %d/%d."\
                      % (j, histogramBins))
            outputfile.write("\n%d" % histogramData)
    
    inputfile.close()
    outputfile.close()


"""
Convert .phu files to .txt files using readPHU, user defines inputPath to files
"""
def PHUtoTXT(inputPath):
    for folder in glob.glob(inputPath):
        for file in glob.glob(folder + "*.phu"):
            # gather info recorded in filename - concentration, wavelength, and trial number
            trialInfo = file[-14:-4]
            # check to make sure it makes sense
            print(trialInfo)
            # generate new filename for processed data file
            newFileName = os.path.join(file[:-14],trialInfo+".txt")
            # read datafile and gather essential information from ASCII header
            readPHU(file,newFileName)
    
    
"""
Take .txt files, use processData to extract relevant info. Save to dataframe, return.
User defines inputPath, and function grabs all .txt files from within.
"""
def loadDataFiles(basePath):
    # process those data files to get one big ol array of everything
    alldfs = []
    for folder in glob.glob(basePath):
        bigDataArray = []
        for filePath in glob.glob(folder + "/*.txt"):
            fileValues = processData(filePath)
            bigDataArray.append(fileValues)
        
        # # convert to dataframe, return
        df = pd.DataFrame(bigDataArray,columns = ["conc", "wavelength", "trial", "inputRate", "floor", "FWHM", "timeMax", "binCents", "rawHist", "PDF", "noiseFactor", "integTime", "dateTime"])
        alldfs.append(df)
    return alldfs


'''
function for processing data from pre-converted PHU hydraharp files

input file type: .txt
filepath is the only variable going in, as "file"

bin size gives histogram bin size, converted into a time step, which is used to 
generate an array of time values from 0 to 12.5ns out of the bins

from here, data extracted from text file:
    input count rate at SPAD (inputCountRate)
    length of time the data was collected (integrationTime)
    datetime of data collection (collectTime)
    concentration from filename, converted into silica weight concentration
    trial number if needed (I cant currently remember what this is)
    wavelength extracted from filename
    histogram truncated to 12.5ns
    noise floor found by randomly selecting a point early on
    FWHM calculated 
    peak extreme first passage time determined by finding array location of max
    normalized to convert to probability distribution function
    relevant values stored into an array for each file
'''
def processData(file):
    binSize = 0.0020              # bin size of hydraharp, in ns
    timeStep = binSize     # time step of count data, in s
    # create time array
    time = np.arange(0,12.5, timeStep)
    binCenters = time + timeStep/2.
    
    # extract data
    data = np.genfromtxt(file,skip_header = 272, max_rows=65536)
    
    # extract input count rate
    inputCountRate = np.genfromtxt(file,skip_header=127,max_rows=1)[1]
    
    # get integration time
    integrationTime = np.genfromtxt(file,skip_header=16,max_rows=1)[1]/1000
    
    # get time of capture
    collectTime = int(datetime.strptime(str(np.loadtxt(file,skiprows = 7, max_rows=1, dtype=str)[2:6]),"['%b' '%d' '%H:%M:%S' '%Y']").strftime('%Y%m%d%H%M%S'))
    
    # get concentration
    concentration = file[-14:-12]
    # deal with weird names I gave things
    if concentration == '--': # air counts
        concentration = -1.
    elif concentration == '99': # 100% ludox counts filename goof up
        concentration = 1.  # WEIGHT concentration of 100% ludox
    else:
        concentration = float(concentration)/100   # everything else in normal WEIGHT concentrations
    # convert weight concentration to volume concentration
    concentration = concentration * 0.235

    # append trial number to list    
    trial = file[-5]
    if trial == "-":
        trial = float(99)
    elif trial == "m":
        trial = float(95)
    elif trial == "q":
        trial = float(50)
    else:
        trial = float(file[-5])
    
    wavelength = float(file[-9:-6])

    # truncate data
    # cut out anything past 12.5 ns
    rawHistogram = data[:len(time)]
    
    # find data "floor" - pick value way out before peak
    floor = rawHistogram[np.where(time==1)][0]
    

    
    # find FWHM
    halfMaxLo = np.min(np.where(rawHistogram >= np.max(rawHistogram)/2))
    halfMaxHi = np.max(np.where(rawHistogram >= np.max(rawHistogram)/2))
    FWHM = binCenters[halfMaxHi]-binCenters[halfMaxLo]

    # find time of peak 
    maxLocation = np.where(rawHistogram == np.max(rawHistogram))[0]
    timeMax = binCenters[maxLocation][0]
    
    
    # convert to PDF
    PDF = countsToPDF(rawHistogram,binSize)
    
    # get noise parameters    
    noise = noiseCalculator(maxLocation, PDF)
    
    # return a butt-ton of things
    return [concentration, wavelength, trial, inputCountRate, floor, FWHM, timeMax, binCenters, rawHistogram, PDF, noise, integrationTime, collectTime]


''' linear function fitter '''    
def linFit(data):
    lambdas = []
    allEFPTs = []
    allnVals = []
    allSTDVs = []
    allconcs = []
    
    # cycle through each wavelength
    for i in data.wavelength.drop_duplicates().to_numpy():
        # separate out data taken at that wavelength
        dfLam = data[data['wavelength']==i]
        dfLam = dfLam.filter(['conc','timeMax','dateTime'])
        
        # get array of available concentrations, dropping duplicates
        concs = dfLam.conc.drop_duplicates().to_numpy()
        # only use wavelengths for which there are multiple concentration datapoints:
        if len(concs) > 1:
            # only use wavelengths for which there's a water reference point
            if concs[0]==0:
                m, c = curve_fit(y, dfLam.conc.to_numpy(), dfLam.timeMax.to_numpy())
                timeAdjust = m[0]*0 + m[1]
        
                meanEFPTs = []
                conVals = []
                errorEFPTs = []
                nVals = []
    
    
                lambdas.append(i)
    
    
                for j in concs:
                    dfCon = dfLam[dfLam['conc']==j]
                    dfCon = dfCon.filter(['conc','timeMax','dateTime'])
                    nVals.append(len(dfCon.timeMax.to_numpy()))
                    conVals.append(j)
                    meanEFPTs.append(dfCon.timeMax.mean()-timeAdjust)
                    
                    times = dfCon.timeMax.to_numpy()
                    # for t in times:
                    #     conVals.append(j)
                    #     meanEFPTs.append((t-timeAdjust))
                    errorEFPTs.append(dfCon.timeMax.std()/len(times))
                allEFPTs.append(meanEFPTs)
                allnVals.append(nVals)
                allconcs.append(conVals)
                allSTDVs.append(errorEFPTs)
    lambdas = np.array(lambdas)
    
    return [lambdas, allconcs, allEFPTs, allSTDVs, allnVals]

''' 
fit x,y data to gumbel distribution via scipy.optimize curve_fit function
    
prefactor A, peak location mu, scaling factor beta, noise k
'''
def gumbelFit(xpoints,ypoints,A,mu,beta,k):
    # calculate best fits for data
    try:
        # be efficient the first time
        parameters, covariance =  curve_fit(gumbel,xpoints,ypoints,[A,mu,beta,k],bounds=(0.,[np.inf,12.5,12.5,np.inf]), maxfev=5000)
    except:
        # we tried to be efficient but it didn't work. take longer now
        parameters, covariance =  curve_fit(gumbel,xpoints,ypoints,[A,mu,beta,k],bounds=(0.,[np.inf,12.5,12.5,np.inf]), maxfev=20000)
    
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stDevs = np.sqrt(np.diag(covariance))
    # Calculate the residuals
    residuals = ypoints - gumbel(xpoints, *parameters)
    return parameters


''' 
fit x,y data to gumbel sum distribution via scipy.optimize curve_fit function
    
prefactor A1 A2, peak location mu1 mu2, scaling factor b1 b2, noise k
'''
def gumbelSumFit(xpoints,ypoints,A1,A2,mu1,mu2,b1,b2,k):
    # calculate best fits for data
    try:
        # be efficient the first time
        parameters, covariance =  curve_fit(gumbelSum,xpoints,ypoints,[A1,A2,mu1,mu2,b1,b2,k],bounds=(0.,[np.inf,np.inf,12.5,12.5,12.5,12.5,np.inf]),maxfev=5000)  
    except:
        # we tried to be efficient but it didn't work. take longer now
        parameters, covariance =  curve_fit(gumbelSum,xpoints,ypoints,[A1,A2,mu1,mu2,b1,b2,k],bounds=(0.,[np.inf,np.inf,12.5,12.5,12.5,12.5,np.inf]),maxfev=20000)
   
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stDevs = np.sqrt(np.diag(covariance))
    # Calculate the residuals
    residuals = ypoints - gumbelSum(xpoints, *parameters)
    return gumbelSum(xpoints, *parameters), parameters



''' truncated colormap function I stole '''
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



''' plot regular ol histogram '''
def plotHistogram(data,xmin,xmax):
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (ns)",size=16)
    ax.set_ylabel("Raw counts",size=16)
    for i in data.index:
        ax.plot(data.binCents[i], data.rawHist[i])
    ax.tick_params(axis='both', labelsize=14)
    ax.tick_params(which='minor')
    ax.set_xlim(xmin,xmax)
    
    

''' plot PDF version of histogram '''    
def plotPDF(data,xmin,xmax):
    fig, ax = plt.subplots()
    colors = cm.Blues(np.linspace(0.5, 1, len(data)))
    ax.set_xlabel("Time (ns)",size=16)
    ax.set_ylabel("Normalized counts",size=16)
    colorval = 0
    for i in data.index:
        print(data.conc[i],data.timeMax[i])
        ax.plot(data.binCents[i], data.PDF[i],c=colors[colorval])
        plt.vlines(x = data.timeMax[i], ymin = 0.0, ymax = np.max(data.PDF[i].flatten()),linestyle='dashed',color=colors[colorval])
        colorval += 1
    ax.set_xlim(xmin,xmax)
    ax.tick_params(axis='both', labelsize=14)
    plt.savefig('C:\\Users\\ancgo\\Documents\\python-figures\\sample_PDFs.pdf', bbox_inches='tight')


''' 
Take in non-adjusted array of data

Plot, concentration on the x and adjusted EFPT on the y, color = wavelength
'''
def plotPeaksWavelength(data):
    cmap_base = 'autumn_r'
    vmin, vmax = 0.2, 1
    cmap = truncate_colormap(cmap_base, vmin, vmax)

    
    fig, ax = plt.subplots()
    ax.set_xlabel("Volume concentration of silica nanospheres",size=16)
    ax.set_ylabel("Time (ns)",size=16)
    plot = ax.scatter(data.conc, data.timeMax, c = data.wavelength.to_numpy(),cmap=cmap)
    cbar = fig.colorbar(plot)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.get_yaxis().labelpad = -10
    cbar.ax.set_ylabel('Wavelength (nm)',size=14, rotation=90)
    ax.tick_params(axis='both',  labelsize=14)
    plt.savefig('C:\\Users\\ancgo\\Documents\\python-figures\\peaks-vs-concentration.pdf', bbox_inches='tight')
    
    

''' 
Plot one set of predictions

Plot, concentration on the x and EFPT delay on the y, color = wavelength
'''
def plotPredictions(predicts,ymin,ymax):
    # generate figure
    fig,ax = plt.subplots(1, figsize=(5,5))
    
    # make a colormap for the colorbar
    cmap_base = 'plasma'
    vmin, vmax = 0.2, 0.75
    # make colors for the predictions
    cmap = truncate_colormap(cmap_base, vmin, vmax)        
    colors = cm.plasma(np.linspace(0.2,0.75, 2))
    
    # make plot pretty
    plt.tick_params(axis='both', labelsize=14)
    plt.xlabel('$C$',size=16)
    plt.ylabel(r'$t(C) - t(0)$ (ps)',size=16)
    plt.xlim(-0.015,0.28)
    plt.ylim(ymin,ymax)

    # plot predictions
    plt.plot(predicts.conc,1000*predicts.diff_680_dif, ":", c =  colors[0], alpha=1.5*0.62)
    plt.plot(predicts.conc,1000*predicts.diff_830_dif,":", c =  colors[-1], alpha=1.5*0.62)

    # make colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(680, 830), cmap=cmap), ax=ax, location='bottom',ticks=[680, 830], orientation='horizontal', label='Wavelength (nm)', aspect=30, pad=0.15)
    cbar.ax.set_xlabel('Wavelength (nm)',size=16,labelpad=-10)
    cbar.ax.tick_params(labelsize=14)
    
    # save to file
    plt.savefig('C:\\Users\\ancgo\\Documents\\python-figures\\temp_predicts.pdf', bbox_inches='tight')
   
    
''' 
Plot all the predictions with the data

Plot, concentration on the x and EFPT delay on the y, color = wavelength

IoR predictions solid line
Diffusion predictions dotted
Telegraph predictions dashed
'''
def plotPredictionsWithData(predictionsTel, predictionsDiff, adjustedData):
    # generate figure subplots
    fig, ax = plt.subplots(2, figsize=(5,3))
    plt.subplots_adjust(bottom = -1.75, hspace=0.25)
    
    # make the labels pretty - top plot 
    ax[0].set_xlabel('$C$',size=16)
    ax[0].set_ylabel(r'$t(C) - t(0)$ (ps)',size=16)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[0].set_xlim(-0.015,0.25)
    ax[0].set_ylim(-50,200)
    
    # make the labels pretty - bottom plot
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlim(1e-2,0.28)
    ax[1].set_ylim(0.5,1000)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylabel(r'$t(C) - t(0)$ (ps)',size=16)
    ax[1].set_xlabel('$C$',size=16,labelpad=-0.1)
    
    # generate color scheme for predictions and colorbar
    predictColors = cm.plasma(np.linspace(0.2,0.75, 2))
    cmap_base = 'plasma'
    vmin, vmax = 0.2, 0.75
    cmap = truncate_colormap(cmap_base, vmin, vmax)
    
    # plot index of refraction predictions
    ax[0].plot(predictionsDiff.conc, 1000*predictionsDiff.IoR_680_dif, c = predictColors[0])
    ax[0].plot(predictionsDiff.conc, 1000*predictionsDiff.IoR_830_dif, c = predictColors[-1])
    ax[1].plot(predictionsDiff.conc, 1000*predictionsDiff.IoR_680_dif, c = predictColors[0])
    ax[1].plot(predictionsDiff.conc, 1000*predictionsDiff.IoR_830_dif, c = predictColors[-1])

    # plot diffusion predictions
    ax[0].plot(predictionsDiff.conc, 1000*predictionsDiff.diff_680_dif,':', c =  predictColors[0])
    ax[0].plot(predictionsDiff.conc, 1000*predictionsDiff.diff_830_dif, ':', c =  predictColors[-1])
    ax[1].plot(predictionsDiff.conc, 1000*predictionsDiff.diff_680_dif,':', c =  predictColors[0])
    ax[1].plot(predictionsDiff.conc, 1000*predictionsDiff.diff_830_dif, ':', c =  predictColors[-1])

    # plot telegraph predictions
    ax[0].plot(predictionsTel.conc_680, 1000*predictionsTel.tel_680_dif,'--', c =  predictColors[0])
    ax[0].plot(predictionsTel.conc_830, 1000*predictionsTel.tel_830_dif, '--', c =  predictColors[-1])
    ax[1].plot(predictionsTel.conc_680, 1000*predictionsTel.tel_680_dif,'--', c =  predictColors[0])
    ax[1].plot(predictionsTel.conc_830, 1000*predictionsTel.tel_830_dif, '--', c =  predictColors[-1])

    # plot actual data
    for j in range(len(adjustedData)):
        colors = cm.plasma(np.linspace(0.2,0.75, len(adjustedData[j][0])))
        for i in range(len(adjustedData[j][0])):
            # scatter plot each array of concentrations and adjusted EFPTs
            ax[0].scatter(adjustedData[j][1][i], 1000*np.array(adjustedData[j][2][i]), color=colors[i])
            ax[1].scatter(adjustedData[j][1][i], 1000*np.array(adjustedData[j][2][i]), color=colors[i])

    # generate colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(680, 830), cmap=cmap), ax=ax, location='bottom',ticks=[680, 830], orientation='horizontal', label='Wavelength (nm)', aspect=30, pad=0.07)
    cbar.ax.set_xlabel('Wavelength (nm)',size=16,labelpad=-10)
    cbar.ax.tick_params(labelsize=14)
    
    # save plot to file
    plt.savefig('C:\\Users\\ancgo\\Documents\\python-figures\\temp_predictions_with_data.pdf', bbox_inches='tight')
    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    

if __name__ == "__main__":
    
    # read in data files from hydraharp folder, dump info into dataframe
    basePath = "C:/Users/ancgo/Documents/hydraharp/all-data/grouped-data/*/"
    # PHUtoTXT(basePath)
    dfs = loadDataFiles(basePath)
    
    # create empty arrays for data storage
    dfsNoAir = []
    dfsBad = []
    dfsAdjusted = []
    dfsGoodAdjusted = []
    for df in dfs:
        # make dataframes for water, air, and ludox data
        df_noAir = df[df["conc"]>-0.1]
        # remove anything that is particularly noisy
        thresh = 0.02
        df_good = df_noAir[(df_noAir['noiseFactor'] < thresh)]
        df_bad = df_noAir[(df_noAir['noiseFactor'] >= thresh)]

        dfsNoAir.append(df_noAir)
        dfsBad.append(df_bad)
        dfsAdjusted.append(linFit(df_noAir))
        dfsGoodAdjusted.append(linFit(df_good))
    
    
    # load in predictions
    dfPredictions_tel_5e7 = pd.read_csv("C:/Users/ancgo/Documents/tel_predicts_N5e7.csv")
    dfPredictions_diff_5e7 = pd.read_csv("C:/Users/ancgo/Documents/diff_predicts_N5e7.csv")
    
    # plot it all
    plotPredictionsWithData(dfPredictions_tel_5e7,dfPredictions_diff_5e7,dfsAdjusted)
    plotPredictionsWithData(dfPredictions_tel_5e7,dfPredictions_diff_5e7,dfsGoodAdjusted)
