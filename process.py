import os
import json
import numpy as np
import math
import scipy.interpolate as interpolate
from scipy import signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

import pandas as pd
from finderPeaksSignal import peakFinder
import argparse

# Define landmarks
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

def decayEstimation(Peaks, nSelectedPeaks=4):
    slope, b = np.polyfit(np.arange(len(Peaks)), Peaks, 1)
    if slope < 0:
        return slope
    else:
        return 0

def scaling(landmarks, scale='THUMBSIZE'):
    prevScale = []
    newScale = []

    for idx, landmark in enumerate(landmarks):
        if len(landmark) > 0:
            # Check if landmark has enough points
            required_indices = [WRIST, MIDDLE_FINGER_TIP]
            if all(idx < len(landmark) for idx in required_indices):
                wrist, middle_finger_tip = landmark[WRIST], landmark[MIDDLE_FINGER_TIP]
                try:
                    #original scaling method is distance between wrist and middle finger tip
                    #so compute original scaling methods and store it in prevScale
                    dist = math.dist(wrist, middle_finger_tip)
                    prevScale.append(dist)
                except Exception as e:
                    print(f"Error computing distance for frame {idx}: {e}")
                    continue  # Skip this frame

                #if new scaling mehthod is selected, compute the new scaling method
                if scale == 'THUMBSIZE':
                    required_indices = [THUMB_CMC, THUMB_TIP]
                    if all(idx < len(landmark) for idx in required_indices):
                        # thumb_base, thumb_tip = landmark[THUMB_CMC], landmark[THUMB_TIP]
                        try:
                            #compute the size of each phalange
                            dist1 = math.dist(landmark[THUMB_CMC], landmark[THUMB_MCP])
                            dist2 = math.dist(landmark[THUMB_MCP], landmark[THUMB_IP])
                            dist3 = math.dist(landmark[THUMB_IP], landmark[THUMB_TIP])
                            #compute the size of the thumb
                            thumb_size = dist1 + dist2 + dist3
                            newScale.append(thumb_size)
                            # dist = math.dist(thumb_base, thumb_tip)
                            # newScale.append(dist)
                        except Exception as e:
                            print(f"Error computing thumb size for frame {idx}: {e}")
                            newScale.append(prevScale[-1])  # Use previous scale
                    else:
                        # Handle missing thumb landmarks
                        print(f"Missing thumb landmarks in frame {idx}")
                        newScale.append(prevScale[-1])  # Use previous scale
                elif scale == 'INDEXSIZE':
                    required_indices = [INDEX_FINGER_MCP, INDEX_FINGER_PIP,INDEX_FINGER_DIP, INDEX_FINGER_TIP]
                    if all(idx < len(landmark) for idx in required_indices):
                        # index_base, index_tip = landmark[INDEX_FINGER_MCP], landmark[INDEX_FINGER_TIP]
                        try:
                            #compute the size of each phalange
                            dist1 = math.dist(landmark[INDEX_FINGER_MCP], landmark[INDEX_FINGER_PIP])
                            dist2 = math.dist(landmark[INDEX_FINGER_PIP], landmark[INDEX_FINGER_DIP])
                            dist3 = math.dist(landmark[INDEX_FINGER_DIP], landmark[INDEX_FINGER_TIP])
                            #compute the size of the index finger
                            index_size = dist1 + dist2 + dist3
                            newScale.append(index_size)
                            # dist = math.dist(index_base, index_tip)
                            # newScale.append(dist)
                        except Exception as e:
                            print(f"Error computing index finger size for frame {idx}: {e}")
                            newScale.append(prevScale[-1])  # Use previous scale
                    else:
                        # Handle missing index finger landmarks
                        print(f"Missing index finger landmarks in frame {idx}")
                        newScale.append(prevScale[-1])  # Use previous scale
                elif scale == 'PALMSIZE':
                    required_indices = [WRIST, INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP]
                    if all(idx < len(landmark) for idx in required_indices):
                        try:
                            #compute the average distance from the wrist to each finger
                            dist1 = math.dist(landmark[INDEX_FINGER_MCP], landmark[WRIST])
                            dist2 = math.dist(landmark[MIDDLE_FINGER_MCP], landmark[WRIST])
                            dist3 = math.dist(landmark[RING_FINGER_MCP], landmark[WRIST])
                            dist4 = math.dist(landmark[PINKY_MCP], landmark[WRIST])
                            #compute the size of the palm
                            palm_size = (dist1 + dist2 + dist3 +dist4 )/4
                            newScale.append(palm_size)
                        except Exception as e:
                            print(f"Error computing palm size for frame {idx}: {e}")
                            newScale.append(prevScale[-1])
                    else:
                        # Handle missing index finger landmarks
                        print(f"Missing hand base landmarks in frame {idx}")
                        newScale.append(prevScale[-1])  # Use previous scale
                else:
                    newScale.append(prevScale[-1])
            else:
                # Not enough points in landmark, skip this frame
                print(f"Missing required landmarks in frame {idx}")
                continue
        else:
            print(f"Empty landmark data in frame {idx}")
            continue

    # Check if prevScale and newScale are not empty to avoid division by zero
    if len(prevScale) == 0 or len(newScale) == 0:
        # Handle case where scaling cannot be computed
        print("Scaling cannot be computed due to missing landmarks.")
        return 1, 'NOSCALING'  # Return scaling factor of 1 (no scaling)
    else:
        # Divide the new scaling method by the original scaling method
        # to get the scaling factor
        scalingFactor = np.max(median_filter(prevScale, 3)) / np.max(median_filter(newScale, 3))
        return scalingFactor, scale  # Return scaling factor and scaling method

def get_output(up_sample_signal):
    fs = 60
    distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(
        up_sample_signal, fs=fs, minDistance=3, cutOffFrequency=7.5, prct=0.05
    )

    amplitude = []
    peakTime = []
    rmsVelocity = []
    speed = []
    averageOpeningSpeed = []
    averageClosingSpeed = []
    maxOpeningSpeed = []
    maxClosingSpeed = []
    cycleDuration = []
    pauseDuration = []
    hesitationsinMovement = []
    hesitationsinPause = []
   

    Npeaks = len(peaks)
    maxVelocity = np.max(velocity)

    for idx, peak in enumerate(peaks):

        #for some reason, the peakFinder function does not return the opening and closing Peak Index
        # so we need to fill the empty values with the peak index


        #Opening sequence
        x1_start = peak['openingValleyIndex']* (1 / fs)
        y1_start = distance[peak['openingValleyIndex']]
        x1_end = peak['openingPeakIndex']* (1 / fs)
        y1_end = distance[peak['openingPeakIndex']]
        #Closing sequence
        x2_start = peak['closingPeakIndex']* (1 / fs)
        y2_start = distance[peak['closingPeakIndex']]
        x2_end = peak['closingValleyIndex']* (1 / fs)
        y2_end = distance[peak['closingValleyIndex']]
        #peak
        x = peak['peakIndex'] * (1 / fs)
        y = distance[peak['peakIndex']]

        #To identify the heigh of the peak, we create a line between the two points corresponding to the opening and closing. Then we compute the distance between the peak and the line
        #This is a function that can be evaluated at any value of x between x1 and x2
        f = interpolate.interp1d(np.array([x1_start, x2_end]), np.array([y1_start, y2_end]))
        amplitude.append(np.abs(y - f(x)))

        # Root Mean Square Velocity between the opening and closing valleys
        rmsVelocity.append(np.sqrt(np.mean(velocity[peak['openingValleyIndex']:peak['closingValleyIndex']] ** 2)))
        # Speed is the distance between the peak and the line divided by the time between the opening and closing valleys
        speed.append(
            (y - f(x)) / ((x2_end - x1_start) )
        )
        #Opening Speed is the distance between the peak and the line divided by the time between the opening valley and the peak of the opening sequence
        averageOpeningSpeed.append(
            np.abs((y1_end - f(x1_end))) / ((x1_end - x1_start) )
        )
        #closing Speed
        averageClosingSpeed.append(
            np.abs((y2_start - f(x2_start))) / ((x2_end - x2_start) )
        )
        #max Opening Speed
        maxOpeningSpeed.append(
            np.max(velocity[peak['openingValleyIndex']:peak['openingPeakIndex']])
        )
        #max Closing Speed
        maxClosingSpeed.append(
            np.max(np.abs(velocity[peak['closingPeakIndex']:peak['closingValleyIndex']]))
        )
        # Cycle duration
        cycleDuration.append(
            ((peak['closingValleyIndex'] - peak['openingValleyIndex']) * (1 / fs) )
        )
        # Timing
        peakTime.append(peak['peakIndex'] * (1 / fs))

        #pause time 
        if idx < Npeaks - 1:
            pauseDuration.append((peaks[idx + 1]['openingValleyIndex'] - peak['closingValleyIndex']) * (1 / fs))
            if pauseDuration[-1] < 0:
                pauseDuration[-1] = 0

        # calculate hesitations in the openeing-closing movement sequences 
        abs_velocity_segment = np.abs(velocity[peak['openingValleyIndex']+1:peak['closingValleyIndex']-1])
        # Count the number of times the absolute value of velocity crosses the threshold defined by maxVelocity*0.25
        crossings = np.sum((abs_velocity_segment[:-1] < maxVelocity*0.25) & (abs_velocity_segment[1:] >= maxVelocity*0.25)) + \
                    np.sum((abs_velocity_segment[:-1] >= maxVelocity*0.25) & (abs_velocity_segment[1:] < maxVelocity*0.25))
        
        # If the number of crossings is greater than 4, a hesitation occurred
        if crossings > 4:
            hesitationsinMovement.append(1)
        else:
            hesitationsinMovement.append(0)


        # calculate hesitations in the during pauses
        if idx < Npeaks - 1:
            if pauseDuration[-1] > 0.2: #only consider pauses longer than 0.1 seconds 
                abs_velocity_segment = np.abs(velocity[peak['closingValleyIndex']:peaks[idx + 1]['openingValleyIndex']+1])
                # Count the number of times the absolute value of velocity crosses the threshold defined by maxVelocity*0.25
                crossings = np.sum((abs_velocity_segment[:-1] < maxVelocity*0.25) & (abs_velocity_segment[1:] >= maxVelocity*0.25)) + \
                            np.sum((abs_velocity_segment[:-1] >= maxVelocity*0.25) & (abs_velocity_segment[1:] < maxVelocity*0.25))
                
                # If the number of crossings is greater than 4, a hesitation occurred
                if crossings > 4:
                    hesitationsinPause.append(1)
                else:
                    hesitationsinPause.append(0)


    if len(amplitude) == 0:
        print("No peaks detected; cannot compute output parameters.")
        return None

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

    meanMaxOpeningSpeed = np.mean(maxOpeningSpeed)
    stdMaxOpeningSpeed = np.std(maxOpeningSpeed)

    meanMaxClosingSpeed = np.mean(maxClosingSpeed)
    stdMaxClosingSpeed = np.std(maxClosingSpeed)


    meanCycleDuration = np.mean(cycleDuration)
    stdCycleDuration = np.std(cycleDuration)

    meanPauseDuration = np.mean(pauseDuration)
    stdPauseDuration = np.std(pauseDuration)


    # Compute the number of pauses as the number of cycles with a duration greater than 2 times the mean cycle duration
    # and the number of pauses with a duration greater than 2 times the mean pause duration
    numPauses = 0 
    numPauses = sum(1 for duration in cycleDuration if duration > 2 * meanCycleDuration) + sum(1 for duration in pauseDuration if duration > 2 * meanPauseDuration)

    hesitations = np.sum(hesitationsinMovement) + np.sum(hesitationsinPause)

    # Compute the range of cycle duration
    if len(peakTime) > 1:
        rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    else:
        rangeCycleDuration = 0

    #Compute the average frequency as the number of peaks divided by the time between the first and last peak
    frequency = len(peaks) / ((peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex']) * (1 / fs))

    # Initialize decay variables
    rateDecay = np.nan
    amplitudeDecay = np.nan
    velocityDecay = np.nan

    # Check if there are enough peaks to compute decay parameters
    if len(peaks) >= 3:
        n = len(peaks) // 3 # Split the peaks into 3 parts
        if n == 0:
            n = 1  # Ensure at least one peak is selected

        earlyPeaks = peaks[:n]
        latePeaks = peaks[-n:]

        # Ensure earlyPeaks and latePeaks are not empty
        if earlyPeaks and latePeaks and len(earlyPeaks) > 0 and len(latePeaks) > 0:
            # Rate Decay
            earlyDuration = (earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex']) * (1 / fs)
            lateDuration = (latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex']) * (1 / fs)

            earlyRate = len(earlyPeaks) / earlyDuration if earlyDuration != 0 else np.nan
            lateRate = len(latePeaks) / lateDuration if lateDuration != 0 else np.nan

            rateDecay = earlyRate / lateRate if lateRate and lateRate != 0 else np.nan

            # Amplitude Decay
            earlyAmplitude = np.array(amplitude)[:n]
            lateAmplitude = np.array(amplitude)[-n:]
            if np.mean(lateAmplitude) != 0:
                amplitudeDecay = np.mean(earlyAmplitude) / np.mean(lateAmplitude)
            else:
                amplitudeDecay = np.nan

            # Velocity Decay
            earlySpeed = np.array(speed)[:n]
            lateSpeed = np.array(speed)[-n:]
            if np.mean(lateSpeed) != 0:
                velocityDecay = np.mean(earlySpeed) / np.mean(lateSpeed)
            else:
                velocityDecay = np.nan

    # Coefficient of Variation
    cvAmplitude = stdAmplitude / meanAmplitude if meanAmplitude != 0 else np.nan
    cvSpeed = stdSpeed / meanSpeed if meanSpeed != 0 else np.nan
    cvRMSVelocity = stdRMSVelocity / meanRMSVelocity if meanRMSVelocity != 0 else np.nan
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed if meanAverageOpeningSpeed != 0 else np.nan
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed if meanAverageClosingSpeed != 0 else np.nan
    cvMaxOpeningSpeed = stdMaxOpeningSpeed / meanMaxOpeningSpeed if meanMaxOpeningSpeed != 0 else np.nan
    cvMaxClosingSpeed = stdMaxClosingSpeed / meanMaxClosingSpeed if meanMaxClosingSpeed != 0 else np.nan
    cvCycleDuration = stdCycleDuration / meanCycleDuration if meanCycleDuration != 0 else np.nan
    cvPauseDuration = stdPauseDuration / meanPauseDuration if meanPauseDuration != 0 else np.nan


    jsonFinal = {
        "MeanAmplitude": meanAmplitude,
        "StdAmplitude": stdAmplitude,
        "MeanSpeed": meanSpeed,
        "StdSpeed": stdSpeed,
        "MeanRMSVelocity": meanRMSVelocity,
        "StdRMSVelocity": stdRMSVelocity,
        "MeanOpeningSpeed": meanAverageOpeningSpeed,
        "StdOpeningSpeed": stdAverageOpeningSpeed,
        "MeanClosingSpeed": meanAverageClosingSpeed,
        "StdClosingSpeed": stdAverageClosingSpeed,
        "MeanMaxOpeningSpeed": meanMaxOpeningSpeed,
        "StdMaxOpeningSpeed": stdMaxOpeningSpeed,
        "MeanMaxClosingSpeed": meanMaxClosingSpeed,
        "StdMaxClosingSpeed": stdMaxClosingSpeed,
        "MeanCycleDuration": meanCycleDuration,
        "StdCycleDuration": stdCycleDuration,
        # "MeanPauseDuration": meanPauseDuration,
        # "StdPauseDuration": stdPauseDuration,
        "CVAmplitude": cvAmplitude,
        "CVSpeed": cvSpeed,
        "CVRMSVelocity": cvRMSVelocity,
        "CVOpeningSpeed": cvAverageOpeningSpeed,
        "CVClosingSpeed": cvAverageClosingSpeed,
        "CVMaxOpeningSpeed": cvMaxOpeningSpeed,
        "CVMaxClosingSpeed": cvMaxClosingSpeed,
        "CVCycleDuration": cvCycleDuration,
        # "CVPauseDuration": cvPauseDuration,
        "Frequency": frequency,
        "AmplitudeDecay": amplitudeDecay,
        "VelocityDecay": velocityDecay,
        # "RateDecay": rateDecay,
        "RangeCycleDuration": rangeCycleDuration,
        "NumberofPauses": numPauses,
        "numberofHesitations": hesitations,
    }
    return jsonFinal, distance, velocity

def get_fileName(file, outputFolder, scalingMethod):
    baseName = os.path.splitext(file)[0]
    # Ensure scalingMethod is suitable for filenames
    scalingMethodStr = scalingMethod.lower().replace(' ', '_')
    newFileName = f"{baseName}_{scalingMethodStr}.csv"
    return os.path.join(outputFolder, newFileName)

def main():
    # call this funtion as 
    # python process.py --inputFolder GoodQualityJSONFiles --outputFolder output --scalingMethod THUMBSIZE

    parser = argparse.ArgumentParser(description="Process input and output folders and scaling method.")
    parser.add_argument("--inputFolder", type=str, required=True, help="Path to the input folder containing JSON files.")
    parser.add_argument("--outputFolder", type=str, required=True, help="Path to the output folder for results.")
    parser.add_argument("--scalingMethod", type=str, required=True, choices=['THUMBSIZE', 'INDEXSIZE', 'PALMSIZE', 'NOSCALING'], help="Scaling method to use.")
    args = parser.parse_args()

    inputFolder = args.inputFolder
    outputFolder = args.outputFolder
    scalingMethod = args.scalingMethod
    # inputFolder = 'GoodQualityJSONFiles'
    # outputFolder = 'output'  # Ensure this folder exists
    # scalingMethod = 'THUMBSIZE'  # You can change this as needed
    #                              #avaliable options are 'THUMBSIZE', 'INDEXSIZE', 'NOSCALING'
    #                              #'NOSCALING' will preserve the original scaling method (hand size)
    
    # inputFolder = "/Users/diegoguarin/Documents/Github/TimeSeriesClassification/test"
    # outputFolder = "/Users/diegoguarin/Documents/Github/TimeSeriesClassification/allFillesCON_THUMB"
    # scalingMethod = 'THUMBSIZE'
    plotResults = True
    listFiles = os.listdir(inputFolder)

    # Ensure the output directory exists
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for file in listFiles:
        if 'json' in file:
            print(file)
            with open(os.path.join(inputFolder, file), encoding="UTF-8") as f:
                data = json.load(f)

            if 'allLandMarks' in data:
                landMarks = data['allLandMarks']
                linePlotData = data['linePlot']['data']
                linePlotTime = data['linePlot']['time']

                # Check if the first element is empty
                if len(landMarks) > 1 and len(landMarks[0]) == 0:
                    landMarks[0] = landMarks[1]
            elif 'landMarks' in data:  
                landMarks = data['landMarks'][0]
                linePlotData = data['linePlot']['data']
                linePlotTime = data['linePlot']['time']
            else:
                landMarks = []
                linePlotData = []
                linePlotTime = []

            if len(landMarks) == 0:
                print(f'File {file} does not contain landmarks')
                continue
            else:

                try:
                    start_time = linePlotTime[0]
                    end_time = linePlotTime[-1]
                    duration = end_time - start_time

                    # upsample the signal to 60 FPS, the find peaks stage works best at 60 FPS
                    fps = 60
                    time_vector = np.linspace(0, duration, int(duration * fps))
                    up_sample_signal = signal.resample(np.array(linePlotData), len(time_vector))
                    # Compute scaling factor
                    if scalingMethod == 'NOSCALING':
                        outParameters, distance, velocity = get_output(up_sample_signal)
                        actualScalingMethod = 'NOSCALING'
                    else:
                        scalingFactor, actualScalingMethod = scaling(landMarks, scalingMethod)
                        
                        # Scale signal and recompute parameters

                        outParameters, distance, velocity = get_output(up_sample_signal * scalingFactor)
       
                    if outParameters is not None:
                        # Save to CSV
                        cvsFilename = get_fileName(file, outputFolder, actualScalingMethod)
                        pd.DataFrame.from_dict(data=outParameters, orient='index').to_csv(cvsFilename, header=False)
                        if plotResults:
                            plt.figure(figsize=(10, 6))
                            plt.plot(np.arange(len(distance)) / 60, distance)
                            plt.xlabel('Time (s)')
                            plt.ylabel('Distance')
                            plt.title('Distance Signal')
                            plt.grid(True)
                            plt.savefig(cvsFilename.replace('.csv', '.png'))
                            plt.close()
                    
                    else:
                        print(f"Skipping file {file} due to insufficient data.")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    if plotResults:
                            plt.figure(figsize=(10, 6))
                            plt.plot(np.arange(len(np.array(linePlotData))) / 60, np.array(linePlotData))
                            plt.xlabel('Time (s)')
                            plt.ylabel('Distance')
                            plt.title('Distance Signal')
                            plt.grid(True)
                            ax = plt.gca()
                            ax.set_facecolor((1.0, 0.47, 0.42,0.25))
                            actualScalingMethod = scalingMethod
                            cvsFilename = get_fileName(file, outputFolder, actualScalingMethod)
                            plt.savefig(cvsFilename.replace('.csv', '.png'))
                            plt.close()
                    continue

                

if __name__ == "__main__":
    main()