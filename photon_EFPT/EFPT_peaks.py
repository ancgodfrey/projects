# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:11:15 2023

@author: Aileen
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

def PHUtoTXT(inputPath):
    for file in glob.glob(inputPath):
        # gather info recorded in filename - concentration, wavelength, and trial number
        trialInfo = file[-14:-4]
        # check to make sure it makes sense
        print(trialInfo)
        # generate new filename for processed data file
        newFileName = os.path.join(basePath,trialInfo+".txt")
        # read datafile and gather essential information from ASCII header
        readPHU(file,newFileName)
    
    
def loadDataFiles(basePath):
    # process those data files to get one big ol array of everything
    bigDataArray = []
    for filePath in glob.glob(basePath + "/*.txt"):
        fileValues = processData(filePath)
        bigDataArray.append(fileValues)
        
    # # convert to dataframe, return
    df = pd.DataFrame(bigDataArray,columns = ["conc", "wavelength", "trial", "inputRate", "floor", "FWHM", "timeMax", "binCents", "rawHist", "PDF", "integTime", "dateTime"])
    return df


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





''' scattering coefficient for Rayleigh; particle size << light wavelength '''
def rayleighCoef(concc,wvlngth):
    Vtube = 0.000179 # volume of tube, m^3
    d = 22e-9 # diameter of ludox silica spheres, m
    n = 1.47012 # IOR for silica
    Vsphere = (4./3.) * np.pi * (d/2)**3 # volume of silica sphere m^3
    nSi = (concc/100 * Vtube) / Vsphere # number of spheres in tube
    vNumSi = nSi/Vtube
    
    lamda = wvlngth * 1e-9

    # calculate scatternig cross section
    crossSec = (2 * np.pi**5)/3 * (d**6 / (lamda**4)) * ((n**2-1) / (n**2+2))**2 
    # calculate scattering coefficient
    coef = vNumSi * crossSec 
    
    return coef




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
    pSi = 2270 # density of silica, kg/m^3
    pWa = 1000 # density of water, kg/m^3
    pLu = 1290 # density of ludox, kg/m^3

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
    timeMax = binCenters[np.where(rawHistogram == np.max(rawHistogram))][0]
    
    # convert to PDF
    PDF = countsToPDF(rawHistogram,binSize)
    # PDFmax = np.max(PDF)
    # PDFfloor = np.min(PDF)
    
    # return a butt-ton of things
    return [concentration, wavelength, trial, inputCountRate, floor, FWHM, timeMax, binCenters, rawHistogram, PDF, integrationTime, collectTime]


'''
calculate how noisy the data is based on the difference between the peak height 
and data 100 spaces away
'''
def noiseCalculator(data):    
    difference = []
    for x in data.index:
       timeMaxInd = np.where(data.binCents[x]==data.timeMax[x])[0][0]
       difference.append(data.PDF[x][timeMaxInd]/data.PDF[x][timeMaxInd-200])
    return 1/np.array(difference)
       


'''
fit histogram to optimized curve

input: data array for a single histogram
'''
def optimizeFit(data):
    for i in data.index:
        floorvalue = data['floor'][i]
        maxvalue = np.max(data['rawHistogram'][i])
        xvalues = data['binCents'][i]
        yvalues = data['rawHistogram'][i]
        concVals = data['concentration'][i]
        muVals = data['timeMax'][i]
        betaVals = data['FWHM'][i]
        wavelengths = data['wavelength'][i]
        
        
        allFits = []
        allWavelengths = []
        allTimes = []
        allParams = []
        allConcs = []
        allFWHMs = []
        rsqmin = 1
                
        for i in range(len(muVals)):
            m1 = float(muVals[i])
            m2 = m1*1.3
            beta1 = float(betaVals[i])/3.
            beta2 = 10*float(betaVals[i])/3.
            kval = floorvalue[i]
            thresholds = np.linspace(0.05*maxvalue[i],0.95*maxvalue[i],5)
            prefac1 = 5e5
            prefac2 = np.linspace(1e4,1e6,20)
            paramsFinal = np.zeros(7)
            rsq = 100
            
            # initial fit
            try:
                fits, fitparams = gumbelSumFit(xvalues[i],yvalues[i],thresholds[0],prefac1,prefac2[0],m1,m2,beta1,beta2,kval) #calculating sum of squared residuals as parameter for fit quality
                r = yvalues[i] - (fits[0] + fits[1])
                rsq = np.sum(np.square(r))
                paramsFinal = np.array(fitparams)
            except:
                fits = np.zeros(np.array(xvalues[i]).shape)
                fitparams = np.zeros(paramsFinal.shape)
                pass
            
            for prefac in prefac2:
                for threshold in thresholds:
                    try:
                        fits, fitparams = gumbelSumFit(xvalues[i],yvalues[i],threshold,prefac1,prefac,m1,m2,beta1,beta2,kval)
                        #calculating sum of squared residuals as parameter for fit quality
                        r = yvalues[i] - fits
                        rsq = np.sum(np.square(r))
                    except:
                        fits = np.zeros(np.array(xvalues[i]).shape)
                        fitparams = np.zeros(paramsFinal.shape)
                        pass
                    
                    if rsq < rsqmin and fitparams[1]>0 and fitparams[3]>0 and fitparams[5]>0 :
                        paramsFinal = fitparams
                        rsqFinal = rsq
            
        allFits.append(fits)
        allParams.append(paramsFinal)
        allWavelengths.append(wavelengths[i])
        allTimes.append(muVals[i])
        allFWHMs.append(betaVals[i])
        allConcs.append(concVals[i])
        
    allValues = np.array([np.array(allConcs),np.array(allTimes),np.array(allFWHMs),np.array(allWavelengths)])
    return allParams, allFits, allValues


'''adjust peaks by wavelength to be relative to water values'''
def adjustPeaksWavelength(data):
    lambdas = []
    allEFPTs = []
    allSTDVs = []
    allNVals = []
    allconcs = []
    # pass through all available wavelength values
    for i in data.wavelength.drop_duplicates().to_numpy():
        # truncate dataframe for relevant wavelength with relevant data
        dfLam = data[data['wavelength']==i]
        dfLam = dfLam.filter(['conc','timeMax','dateTime'])
    
        # create storage for values
        meanEFPTs = []
        stdevEFPTs = []
        NVals = []
        # create array of all available concentrations for this wavelength
        concs = dfLam.conc.drop_duplicates().to_numpy()
        # if there's a value for pure water
        if concs[0]==0:
            # add this wavelength to array
            lambdas.append(i)
            # find water entries for this wavelength
            dfZero = dfLam[dfLam['conc']==0]
            # find mean EFPT value through water at this wavelength, this will
            # be the value to use for adjusting everything else
            timeAdjust = dfZero.timeMax.mean()
            
            # pass through all availalbe concentrations for this wavelength
            for j in concs:
                # truncate dataframe to this concentration with relevant data
                dfCon = dfLam[dfLam['conc']==j]
                dfCon = dfCon.filter(['conc','timeMax','dateTime'])
                # append values to storage arrays
                NVals.append(len(dfCon.timeMax.to_numpy()))
                meanEFPTs.append((dfCon.timeMax.mean()-timeAdjust))
                stdevEFPTs.append((dfCon.timeMax.std()))
            # append all values for this wavelength to storage arrays
            allEFPTs.append(meanEFPTs)
            allSTDVs.append(stdevEFPTs)
            allNVals.append(NVals)
            allconcs.append(concs)

    # return array containing wavelengths, adjusted EFPTs, standard deviations, N values, and concentrations
    return [lambdas, allconcs, allEFPTs, allSTDVs, allNVals]




'''generate a line for the index of refraction passage time'''
def generateIoRLine(concs, length):
    # create array of indices of refraction for each concentration
    IoRvals = (concs/100 * 1.47012) + (100-concs)/100 * 1.33
    # calculate the speed of light in these media
    speedLight = 2.9979e8 / IoRvals
    
    # create array of "baseline differences" i.e. projected difference in EFPT
    # for the index-averaged media
    baselineDif = []
    for i in range(len(speedLight)):
        baselineDif.append((length/speedLight[i] - length/speedLight[0])*1e9)
        
    # return array of both the EFPTs and baseline differences
    return np.array(length/speedLight, baselineDif)

    

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



''' plot histogram with fitted fit '''
def plotFit(xdata,ydata,fitxdata,fitydata,xmin,xmax):
    fig, ax = plt.subplots(figsize=(15,10))
    colors = cm.rainbow(np.linspace(0, 1, len(xdata)))
    ax.set_title("Extreme FPT histogram with gumbel sum fit")
    for i in range(len(xdata)):
        ax.semilogy(xdata[i], ydata[i], label='data', c=colors[i])
        ax.semilogy(xdata[i], fitydata[i], '--', c=colors[i], label='fit')
    ax.set_xlabel("Time (ns)",size=20)
    ax.set_ylabel("Counts",size=20)
    ax.set_xlim(xmin,xmax)
    ax.tick_params(axis='both', which='both', labelsize=18)
    plt.savefig('C:\\Users\\ancgo\\Documents\\python-figures\\gumbel-sum-fit-data.pdf')
    


''' plot regular ol histogram '''
def plotHistogram(data,xmin,xmax):
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (ns)",size=16)
    ax.set_ylabel("Raw counts",size=16)
    for i in data.index:
        ax.plot(data.binCents[i], data.rawHistogram[i])
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


''' linear equation '''    
def y(x,A,B):
    return A*x + B



''' linear function fitter '''    
def linFit(data):
    lambdas = []
    allEFPTs = []
    allnVals = []
    allconcs = []
    for i in data.wavelength.drop_duplicates().to_numpy():
        dfLam = data[data['wavelength']==i]
        dfLam = dfLam.filter(['conc','timeMax','dateTime'])

        concs = dfLam.conc.drop_duplicates().to_numpy()
        if len(concs) > 1:
            m, c = curve_fit(y, dfLam.conc.to_numpy(), dfLam.timeMax.to_numpy())
            timeAdjust = m[0]*0 + m[1]
    
            meanEFPTs = []
            errorEFPTs = []
            nVals = []
    
            if concs[0]==0:
    
                lambdas.append(i)
    
    
                for j in concs:
                    dfCon = dfLam[dfLam['conc']==j]
                    dfCon = dfCon.filter(['conc','timeMax','dateTime'])
                    nVals.append(len(dfCon.timeMax.to_numpy()))
        
                    meanEFPTs.append(1000*(dfCon.timeMax.mean()-timeAdjust))
                    errorEFPTs.append(1000*(dfCon.timeMax.std()))
                allEFPTs.append(meanEFPTs)
                allnVals.append(nVals)
                allconcs.append(concs/100)
    
    lambdas = np.array(lambdas)
    
    return np.array([lambdas, allEFPTs, allnVals, allconcs])



''' 
Take in array of [wavelengths, concs for each wavelength, efpts for each wavelength, 
                  standard devs for all wavelengths, and N vals for all wavelengths]

Plot that shit, concentration on the x and adjusted EFPT on the y, color = wavelength
'''
def plotPeaksWavelengthScaled(adjustedData):
    # generate colormap, truncated for visual helpfulness
    cmap_base = 'plasma_r'
    vmin, vmax = 0.3, 1
    cmap = truncate_colormap(cmap_base, vmin, vmax)
    colors = cm.plasma_r(np.linspace(0.3, 1, len(adjustedData[0])))

    # generate figure and axes
    fig, ax = plt.subplots(figsize=(6,4.5))
    ax.set_xlabel('C',size=16)
    ax.set_ylabel(r'$t(C) - t(0)$ (ps)',size=16)

    # pass through each wavelength available
    for i in range(len(adjustedData[0])):
        # scatter plot each array of concentrations and adjusted EFPTs
        ax.scatter(adjustedData[1][i], adjustedData[2][i], color=colors[i])

    # create colorbar for wavelength color
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(680, 830), cmap=cmap), ax=ax, location='top', orientation='horizontal', label='Wavelength (nm)')
    cbar.ax.set_xlabel('Wavelength (nm)',size=14)
    cbar.ax.tick_params(labelsize=12)
    
    # other settings
    ax.tick_params(axis='both', labelsize=14)

    # save figure
    plt.savefig('C:\\Users\\ancgo\\Documents\\python-figures\\alldata.pdf', bbox_inches='tight')
    

''' 
Take in non-adjusted array of data

Plot that shit, concentration on the x and adjusted EFPT on the y, color = wavelength
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
Take in non-adjusted array of data

Plot that shit, concentration on the x and EFPT width on the y, color = wavelength
'''
def plotWidths(data):
    cmap_base = 'autumn_r'
    vmin, vmax = 0.2, 1
    cmap = truncate_colormap(cmap_base, vmin, vmax)
    
    
    fig, ax = plt.subplots(figsize=(15,10))
    ax.set_title("FPT peak width vs concentration by wavelength",size=20)
    ax.set_xlabel("Volume concentration of silica nanospheres",size=30)
    ax.set_ylabel("Width (ns)",size=30)
    plot = ax.scatter(data.concentration, data.FWHM, c = data.wavelength.to_numpy(), cmap=cmap)
    cbar = fig.colorbar(plot)
    cbar.ax.tick_params(labelsize=25)
    ax.tick_params(axis='both', length=12, labelsize=25)
    ax.tick_params(which='minor', length=6)
    
   
   
''' 
Now things get ugly
'''
def plotPredictionsAndData(predictionsDiff, predictionsTel, adjustedData):
    cmap_base = 'plasma'
    vmin, vmax = 0.2, 0.75
    cmap = truncate_colormap(cmap_base, vmin, vmax)
    
    fig, ax = plt.subplots()
    ax.set_xlabel('C',size=16)
    ax.set_ylabel(r'$t(C) - t(0)$ (ps)',size=16)
    
    
    ax.plot(predictionsDiff.concc, 1000*predictionsDiff.IoR_dif, c = 'darkturquoise')
    
    wavelengths, EFPTs, errors, nvals, concs = adjustPeaksWavelength(data)
    colors = cm.plasma(np.linspace(0.2,0.75, len(wavelengths)))
    
    a = 0.62
    for i in range(len(wavelengths)):
            EFPTs[i] = np.array(EFPTs[i])
            ax.scatter(concs[i], EFPTs[i], color=colors[i], alpha=a)
    
    ax.plot(predictionsDiff.concc, 1000*predictionsDiff.diff680mu0_dif, '--', c = colors[0], alpha=1.5*a)
    ax.plot(predictionsDiff.concc, 1000*predictionsDiff.diff830mu0_dif,'--', c =  colors[-1], alpha=1.5*a)
    
    ax.plot(predictionsTel.concc2, 1000*predictionsTel.t_680mu00_dif, '-.', c =  colors[0], alpha=1.5*a)
    ax.plot(predictionsTel.concc3, 1000*predictionsTel.t_830mu00_dif,'-.', c =  colors[-1], alpha=1.5*a)
    
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim(7.5e-3,0.25)
    ax.set_ylim(1e-1,5e4)
    ax.set_xscale('log')
    ax.set_yscale('log')

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(680, 830), cmap=cmap), ax=ax, location='top', orientation='horizontal', label='Wavelength (nm)')
    cbar.ax.set_xlabel('Wavelength (nm)',size=14)
    cbar.ax.tick_params(labelsize=12)
    
    plt.savefig('C:\\Users\\ancgo\\Documents\\python-figures\\diff-losslesstel-predictions-alldata-ps.pdf', bbox_inches='tight')


''' 
Now things get REAL ugly
'''
def plotPredictionsAndDataTwoFigs(predictionsDiff, predictionsTel, predictionsTel_new, adjustedData):

    cmap_base = 'plasma'
    vmin, vmax = 0.2, 0.75
    cmap = truncate_colormap(cmap_base, vmin, vmax)

    wavelengths, EFPTs, nvals, concs = linFit(data)
    colors = cm.plasma(np.linspace(0.2,0.75, len(wavelengths)))

    fig, ax = plt.subplots(2, figsize=(5,3))
    plt.subplots_adjust(bottom = -1.75, hspace=0.25)
    a = 0.62
    ax[0].set_xlabel('C',size=16)
    ax[1].set_xlabel('C',size=16,labelpad=-5)
    ax[0].set_ylabel(r'$t(C) - t(0)$ (ps)',size=16)
    ax[1].set_ylabel(r'$t(C) - t(0)$ (ps)',size=16)

    ax[0].plot(predictionsDiff.conccs, 1000*predictionsDiff.FPT_IoR_dif, c = 'darkturquoise')
    for i in range(len(wavelengths)):
        EFPTs[i] = np.array(EFPTs[i])
        ax[0].scatter(concs[i], EFPTs[i], color=colors[i])

    ax[0].plot(predictionsDiff.conccs2, 1000*predictionsDiff.FPT_680_lossy_dif, ':', c = colors[0], alpha=1.5*a)
    ax[0].plot(predictionsDiff.conccs2, 1000*predictionsDiff.FPT_830_lossy_dif,':', c =  colors[-1], alpha=1.5*a)
    ax[0].plot(predictionsTel_new.concc7, 1000*predictionsTel_new.redone_830_dif,'--', c =  colors[-1], alpha=1.5*a)
    ax[0].plot(predictionsTel_new.concc6, 1000*predictionsTel_new.redone_680_dif, '--', c =  colors[0], alpha=1.5*a)
    ax[1].plot(predictionsDiff.conccs, 1000*predictionsDiff.FPT_IoR_dif, c = 'darkturquoise')

    for i in range(len(wavelengths)):
            EFPTs[i] = np.array(EFPTs[i])
            ax[1].scatter(concs[i], EFPTs[i], color=colors[i], alpha=a)

    ax[1].plot(predictionsDiff.conccs2, 1000*predictionsDiff.FPT_680_lossy_dif, ':', c = colors[0], alpha=1.5*a)
    ax[1].plot(predictionsDiff.conccs2, 1000*predictionsDiff.FPT_830_lossy_dif,':', c =  colors[-1], alpha=1.5*a)
    ax[1].plot(predictionsTel_new.concc7, 1000*predictionsTel_new.redone_830_dif,'--', c =  colors[-1], alpha=1.5*a)
    ax[1].plot(predictionsTel_new.concc6, 1000*predictionsTel_new.redone_680_dif, '--', c =  colors[0], alpha=1.5*a)
    ax[1].scatter(predictionsTel_new.concc7.dropna().to_numpy()[-1], 1000*predictionsTel_new.redone_830_dif.dropna().to_numpy()[-1],marker='o', color = (0.1,0.1,0.1,0),edgecolors =  colors[-1])
    ax[1].scatter(predictionsTel_new.concc6.dropna().to_numpy()[-1], 1000*predictionsTel_new.redone_680_dif.dropna().to_numpy()[-1],marker='o', color = (0.1,0.1,0.1,0), edgecolors = colors[0])

    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].set_xlim(-2e-2,0.26)
    ax[0].set_ylim(-15,130)
    ax[1].set_xlim(7.5e-3,0.275)
    ax[1].set_ylim(4e-1,2e5)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(680, 830), cmap=cmap), ax=ax, location='bottom',ticks=[680, 830], orientation='horizontal', label='Wavelength (nm)', aspect=30, pad=0.07)

    #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])
    cbar.ax.set_xlabel('Wavelength (nm)',size=16,labelpad=-10)
    cbar.ax.tick_params(labelsize=14)
    plt.savefig('C:\\Users\\ancgo\\Documents\\python-figures\\lossy-N12-only-new.pdf', bbox_inches='tight')
    

''' 
Thought it couldnt get worse? Think again
'''
def plotPredictionsLowNHighNTwoFigs(predictionsDiff, predictionsTel, predictionsTel_new, adjustedData):

    cmap_base = 'plasma'
    vmin, vmax = 0.2, 0.75
    cmap = truncate_colormap(cmap_base, vmin, vmax)

    wavelengths, EFPTs, nvals, concs = linFit(data)
    colors = cm.plasma(np.linspace(0.2,0.75, len(wavelengths)))

    fig, ax = plt.subplots(2, figsize=(5,3))
    plt.subplots_adjust(bottom = -1.75, hspace=0.25)
    a = 0.62
    ax[0].set_xlabel('C',size=16)
    ax[1].set_xlabel('C',size=16,labelpad=-5)
    ax[0].set_ylabel(r'$t(C) - t(0)$ (ps)',size=16)
    ax[1].set_ylabel(r'$t(C) - t(0)$ (ps)',size=16)

    ax[0].plot(predictionsDiff.conccs, 1000*predictionsDiff.FPT_IoR_dif, c = 'darkturquoise')
    for i in range(len(wavelengths)):
        EFPTs[i] = np.array(EFPTs[i])
        ax[0].scatter(concs[i], EFPTs[i], color=colors[i])

    ax[0].plot(predictionsDiff.conccs2, 1000*predictionsDiff.FPT_680_lossy_dif, ':', c = colors[0], alpha=1.5*a)
    ax[0].plot(predictionsDiff.conccs2, 1000*predictionsDiff.FPT_830_lossy_dif,':', c =  colors[-1], alpha=1.5*a)
    ax[0].plot(predictionsDiff.conccs2, 1000*predictionsDiff.lowN_680_lossy_dif, ':', c = colors[0], alpha=1.5*a)
    ax[0].plot(predictionsDiff.conccs2, 1000*predictionsDiff.lowN_830_lossy_dif,':',c =  colors[-1], alpha=1.5*a)
    ax[0].plot(predictionsTel_new.concc7, 1000*predictionsTel_new.redone_830_dif,'-', c =  colors[-1], alpha=1.5*a)
    ax[0].plot(predictionsTel_new.concc6, 1000*predictionsTel_new.redone_680_dif, '-', c =  colors[0], alpha=1.5*a)
    ax[0].plot(predictionsTel_new.concc01, 1000*predictionsTel_new.lowN_lossy_830_dif,'--', c =  colors[-1], alpha=1.5*a)
    ax[0].plot(predictionsTel_new.concc0, 1000*predictionsTel_new.lowN_lossy_680_dif, '--', c =  colors[0], alpha=1.5*a)
    ax[1].plot(predictionsDiff.conccs, 1000*predictionsDiff.FPT_IoR_dif, c = 'darkturquoise')

    for i in range(len(wavelengths)):
            EFPTs[i] = np.array(EFPTs[i])
            ax[1].scatter(concs[i], EFPTs[i], color=colors[i], alpha=a)

    ax[1].plot(predictionsDiff.conccs2, 1000*predictionsDiff.FPT_680_lossy_dif, '-.', c = colors[0], alpha=1.5*a)
    ax[1].plot(predictionsDiff.conccs2, 1000*predictionsDiff.FPT_830_lossy_dif,'-.', c =  colors[-1], alpha=1.5*a)
    ax[1].plot(predictionsDiff.conccs2, 1000*predictionsDiff.lowN_680_lossy_dif, ':', c = colors[0], alpha=1.5*a)
    ax[1].plot(predictionsDiff.conccs2, 1000*predictionsDiff.lowN_830_lossy_dif,':', c =  colors[-1], alpha=1.5*a)
    ax[1].plot(predictionsTel_new.concc7, 1000*predictionsTel_new.redone_830_dif,'-', c =  colors[-1], alpha=1.5*a)
    ax[1].plot(predictionsTel_new.concc6, 1000*predictionsTel_new.redone_680_dif, '-', c =  colors[0], alpha=1.5*a)
    ax[1].plot(predictionsTel_new.concc01, 1000*predictionsTel_new.lowN_lossy_830_dif,'--', c =  colors[-1], alpha=1.5*a)
    ax[1].plot(predictionsTel_new.concc0, 1000*predictionsTel_new.lowN_lossy_680_dif, '--', c =  colors[0], alpha=1.5*a)
    ax[1].scatter(predictionsTel_new.concc3.dropna().to_numpy()[-1], 1000*predictionsTel_new.t_830mu00_dif.dropna().to_numpy()[-1],marker='o', color = (0.1,0.1,0.1,0),edgecolors =  colors[-1])
    ax[1].scatter(predictionsTel_new.concc2.dropna().to_numpy()[-1], 1000*predictionsTel_new.t_680mu00_dif.dropna().to_numpy()[-1],marker='o', color = (1,1,1,0), edgecolors = colors[0])

    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].set_xlim(-2e-2,0.26)
    ax[0].set_ylim(-15,130)
    ax[1].set_xlim(2.5e-3,0.275)
    ax[1].set_ylim(4e-1,2e5)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(680, 830), cmap=cmap), ax=ax, location='bottom',ticks=[680, 830], orientation='horizontal', label='Wavelength (nm)', aspect=30, pad=0.07)

    #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])
    cbar.ax.set_xlabel('Wavelength (nm)',size=16,labelpad=-10)
    cbar.ax.tick_params(labelsize=14)
    plt.savefig('C:\\Users\\ancgo\\Documents\\python-figures\\lossy-N12-N3.pdf', bbox_inches='tight')


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    

if __name__ == "__main__":
    
    basePath = "C:/Users/ancgo/Documents/hydraharp/all-data/"
    
    PHUtoTXT("C:/Users/ancgo/Documents/hydraharp/PHU-files-new/**/*.phu")

    # # read in data files
    # for file in glob.glob("C:/Users/ancgo/Documents/hydraharp/PHU-files-new/**/*.phu"):
    #     # gather info recorded in filename - concentration, wavelength, and trial number
    #     trialInfo = file[-14:-4]
    #     # check to make sure it makes sense
    #     print(trialInfo)
    #     # generate new filename for processed data file
    #     newFileName = os.path.join(basePath,trialInfo+".txt")
    #     # read datafile and gather essential information from ASCII header
    #     readPHU(file,newFileName)
    
    df = loadDataFiles(basePath)
    
    # # process those data files to get one big ol array of everything
    # bigDataArray = []
    # for filePath in glob.glob(basePath + "/*.txt"):
    #     fileValues = processData(filePath)
    #     bigDataArray.append(fileValues)
        
    # # # convert to dataframe
    # df = pd.DataFrame(bigDataArray,columns = ["conc", "wavelength", "trial", "inputRate", "floor", "FWHM", "timeMax", "binCents", "rawHist", "PDF", "integTime", "dateTime"])
    
    # # add in "noise factor" comparing peak of histogram to value 50 steps left of it
    df.insert(11,'noiseFactor',noiseCalculator(df))
    
    # make dataframes for water, air, and ludox data
    df_noAir = df[df["conc"]>-1]
    df_noAir = df_noAir[df_noAir['timeMax']>5.1]
    df_noAir = df_noAir[df_noAir['timeMax']<6.0]
    df_noAir = df_noAir[(df_noAir['inputRate'] > 2e4) & (df_noAir['noiseFactor'] < 3e-4)]
    
    adjust = adjustPeaksWavelength(df_noAir)
    plotPeaksWavelengthScaled(adjust)
    
    dfAir = df[df["conc"]<0]
    dfAir = dfAir[dfAir["timeMax"]<5.0]
    
    dfWtr = df_noAir[df_noAir["conc"]==0]
    dfLdx = df_noAir[df_noAir["conc"]>0]

    
    # # load in predictions, add column I should have put into csv file to begin with
    # dfPredictions = pd.read_csv("C:/Users/ancgo/Documents/prediction_values_scaled.csv")
    # dfPredictions_diff = pd.read_csv("C:/Users/ancgo/Documents/diffPredicts.csv")
    # dfPredictions_tel = pd.read_csv("C:/Users/ancgo/Documents/telPredicts.csv")
    # dfPredictions_tel_new = pd.read_csv("C:/Users/ancgo/Documents/telPredicts_extranew.csv")
    # dfPredictions_diff_new = pd.read_csv("C:/Users/ancgo/Documents/diffPredicts_Lossless.csv")