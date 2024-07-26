import os
import json
import numpy as np
import math
import scipy.interpolate as interpolate
import pandas as pd
from finderPeaksSignal import peakFinder


#define landmarks 

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def scaling(landmarks, scale='THUMBSIZE'):
    
    prevScale = []
    newScale = []

    for landmark in landmarks:

        wrist, middle_finger_tip = landmark[WRIST], landmark[MIDDLE_FINGER_TIP]
        dist = math.dist(wrist, middle_finger_tip)
        prevScale.append(dist)

        if scale == 'THUMBSIZE':
            thumb_base, thumb_tip = landmark[THUMB_CMC], landmark[THUMB_TIP]
            dist = math.dist(thumb_base, thumb_tip)
            newScale.append(dist)
        elif scale == 'INDEXSIZE':
            index_base, index_tip = landmark[INDEX_FINGER_MCP], landmark[MIDDLE_FINGER_TIP]
            dist = math.dist(index_base, index_tip)
            newScale.append(dist)
        else:
            newScale.append(prevScale[-1])

    #factor used to adjust scale
    return np.max(prevScale)/np.max(newScale)

def get_output(up_sample_signal):
    distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(up_sample_signal, fs=60,
                                                                                         minDistance=3,
                                                                                         cutOffFrequency=7.5, prct=0.05)

    amplitude = []
    peakTime = []
    rmsVelocity = []
    speed = []
    averageOpeningSpeed = []
    averageClosingSpeed = []
    cycleDuration = []

    for idx, peak in enumerate(peaks):
        # Height measures
        x1 = peak['openingValleyIndex']
        y1 = distance[peak['openingValleyIndex']]

        x2 = peak['closingValleyIndex']
        y2 = distance[peak['closingValleyIndex']]

        x = peak['peakIndex']
        y = distance[peak['peakIndex']]

        f = interpolate.interp1d(np.array([x1, x2]), np.array([y1, y2]))

        amplitude.append(y - f(x))

        # Opening Velocity
        rmsVelocity.append(np.sqrt(np.mean(velocity[peak['openingValleyIndex']:peak['closingValleyIndex']] ** 2)))


        speed.append( (y - f(x)) / ((peak['closingValleyIndex']- peak['openingValleyIndex'])* (1 / 60)))
        averageOpeningSpeed.append((y - f(x)) / ((peak['peakIndex'] - peak['openingValleyIndex']) * (1 / 60)))
        averageClosingSpeed.append((y - f(x)) / ((peak['closingValleyIndex'] - peak['peakIndex']) * (1 / 60)))
        cycleDuration.append((peak['closingValleyIndex'] - peak['openingValleyIndex'])* (1 / 60))
        # timming
        peakTime.append(peak['peakIndex'] * (1 / 60))

    meanAmplitude = np.mean(amplitude)
    stdAmplitude = np.std(amplitude)

    meanSpeed = np.mean(speed)
    stdSpeed = np.std(speed)

    meanRMSVelocity = np.mean(rmsVelocity)
    stdRMSVelocity = np.std(rmsVelocity)
    meanAverageOpeningSpeed = np.mean(averageOpeningSpeed)
    stdAverageOpeningSpeed = np.std(averageOpeningSpeed)
    meanAverageClosingSpeed = np.mean(averageClosingSpeed)
    stdAverageClosingSpeed = np.std(averageClosingSpeed)

    meanCycleDuration = np.mean(cycleDuration)
    stdCycleDuration = np.std(cycleDuration)
    rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    rate = len(peaks) / (peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex']) / (1 / 60)

    earlyPeaks = peaks[:len(peaks) // 2]
    latePeaks = peaks[-len(peaks) // 2:]
    # amplitudeDecay = np.mean(distance[:len(peaks) // 3]) / np.mean(distance[-len(peaks) // 3:])
    # velocityDecay = np.sqrt(
    #     np.mean(velocity[earlyPeaks[0]['openingValleyIndex']:earlyPeaks[-1]['closingValleyIndex']] ** 2)) / np.sqrt(
    #     np.mean(velocity[latePeaks[0]['openingValleyIndex']:latePeaks[-1]['closingValleyIndex']] ** 2))
    rateDecay = (len(earlyPeaks) / ((earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex']) / (1 / 60))) / (
                        len(latePeaks) / (
                        (latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex']) / (1 / 60)))

    amplitudeDecay = np.array(amplitude)[:len(amplitude)//2].mean() / np.array(amplitude)[len(amplitude)//2:].mean()
    velocityDecay = np.array(rmsVelocity)[:len(rmsVelocity)//2].mean() / np.array(rmsVelocity)[len(rmsVelocity)//2:].mean()



    cvAmplitude = stdAmplitude / meanAmplitude
    cvSpeed = stdSpeed / meanSpeed
    cvCycleDuration = stdCycleDuration / meanCycleDuration
    cvRMSVelocity = stdRMSVelocity / meanRMSVelocity
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed

    jsonFinal = {
            "MeanAmplitude": meanAmplitude,
            "StdAmplitude": stdAmplitude,
            "MeanSpeed": meanSpeed,
            "StdSpeed": stdSpeed,
            "MeanRMSVelocity": meanRMSVelocity,
            "StdRMSVelocity": stdRMSVelocity,
            "MeanOpeningSpeed": meanAverageOpeningSpeed,
            "stdOpeningSpeed": stdAverageOpeningSpeed,
            "meanClosingSpeed": meanAverageClosingSpeed,
            "stdClosingSpeed": stdAverageClosingSpeed,
            "meanCycleDuration": meanCycleDuration,
            "stdCycleDuration": stdCycleDuration,
            "rangeCycleDuration": rangeCycleDuration,
            "rate": rate,
            "amplitudeDecay": amplitudeDecay,
            "velocityDecay": velocityDecay,
            "rateDecay": rateDecay,
            "cvAmplitude": cvAmplitude,
            "cvCycleDuration": cvCycleDuration,
            "cvSpeed": cvSpeed,
            "cvRMSVelocity" : cvRMSVelocity,
            "cvOpeningSpeed": cvAverageOpeningSpeed,
            "cvClosingSpeed": cvAverageClosingSpeed
    }
    return jsonFinal

def get_fileName(file, outputFolder):

    return os.path.join(outputFolder, os.path.splitext(file)[0] + '.csv') 

def main():

    inputFolder = 'data'
    listFiles = os.listdir(inputFolder)

    for file in listFiles:

        f = open(os.path.join(inputFolder,file))
        data = json.load(f)
        f.close()

        if 'allLandMarks' in data:
            landMarks = data['allLandMarks']
            linePlotData = data['linePlot']['data']
            linePlotTime = data['linePlot']['time']
        elif 'landMarks' in data:
            landMarks = data['landMarks'][0]
            linePlotData = data['linePlot']['data']
            linePlotTime = data['linePlot']['time']
        else:
            landMarks = []
            linePlotData = []
            linePlotTime = []
        
        if len(landMarks)==0:
            print(f'File {file} does not contain landmarks')
            continue
        else:
            #compute scaling factor
            scalingFactor = scaling(landMarks)
            #scale signal and recompute parameters
            outParameters = get_output(np.array(linePlotData)*scalingFactor)
            #save to csv
            cvsFilename = get_fileName(file, 'output')
            pd.DataFrame.from_dict(data=outParameters, orient='index').to_csv(cvsFilename, header=False)


if __name__ == "__main__":
    main()
            



