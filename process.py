import os
import json
import numpy as np
import math
import scipy.interpolate as interpolate
from scipy.ndimage import median_filter

import pandas as pd
from finderPeaksSignal import peakFinder

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
                    dist = math.dist(wrist, middle_finger_tip)
                    prevScale.append(dist)
                except Exception as e:
                    print(f"Error computing distance for frame {idx}: {e}")
                    continue  # Skip this frame

                if scale == 'THUMBSIZE':
                    required_indices = [THUMB_CMC, THUMB_TIP]
                    if all(idx < len(landmark) for idx in required_indices):
                        thumb_base, thumb_tip = landmark[THUMB_CMC], landmark[THUMB_TIP]
                        try:
                            dist = math.dist(thumb_base, thumb_tip)
                            newScale.append(dist)
                        except Exception as e:
                            print(f"Error computing thumb size for frame {idx}: {e}")
                            newScale.append(prevScale[-1])  # Use previous scale
                    else:
                        # Handle missing thumb landmarks
                        print(f"Missing thumb landmarks in frame {idx}")
                        newScale.append(prevScale[-1])  # Use previous scale
                elif scale == 'INDEXSIZE':
                    required_indices = [INDEX_FINGER_MCP, MIDDLE_FINGER_TIP]
                    if all(idx < len(landmark) for idx in required_indices):
                        index_base, index_tip = landmark[INDEX_FINGER_MCP], landmark[MIDDLE_FINGER_TIP]
                        try:
                            dist = math.dist(index_base, index_tip)
                            newScale.append(dist)
                        except Exception as e:
                            print(f"Error computing index finger size for frame {idx}: {e}")
                            newScale.append(prevScale[-1])  # Use previous scale
                    else:
                        # Handle missing index finger landmarks
                        print(f"Missing index finger landmarks in frame {idx}")
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
        # Factor used to adjust scale
        scalingFactor = np.max(median_filter(prevScale, 3)) / np.max(median_filter(newScale, 3))
        return scalingFactor, scale  # Return scaling factor and scaling method

def get_output(up_sample_signal):
    distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(
        up_sample_signal, fs=60, minDistance=3, cutOffFrequency=7.5, prct=0.05
    )

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

        speed.append(
            (y - f(x)) / ((peak['closingValleyIndex'] - peak['openingValleyIndex']) * (1 / 60))
        )
        averageOpeningSpeed.append(
            (y - f(x)) / ((peak['peakIndex'] - peak['openingValleyIndex']) * (1 / 60))
        )
        averageClosingSpeed.append(
            (y - f(x)) / ((peak['closingValleyIndex'] - peak['peakIndex']) * (1 / 60))
        )
        cycleDuration.append(
            (peak['closingValleyIndex'] - peak['openingValleyIndex']) * (1 / 60)
        )
        # Timing
        peakTime.append(peak['peakIndex'] * (1 / 60))

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

    meanCycleDuration = np.mean(cycleDuration)
    stdCycleDuration = np.std(cycleDuration)
    if len(peakTime) > 1:
        rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    else:
        rangeCycleDuration = 0
    rate = len(peaks) / ((peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex']) * (1 / 60))

    # Initialize decay variables
    rateDecay = np.nan
    amplitudeDecay = np.nan
    velocityDecay = np.nan

    # Check if there are enough peaks to compute decay parameters
    if len(peaks) >= 3:
        n = len(peaks) // 3
        if n == 0:
            n = 1  # Ensure at least one peak is selected

        earlyPeaks = peaks[:n]
        latePeaks = peaks[-n:]

        # Ensure earlyPeaks and latePeaks are not empty
        if earlyPeaks and latePeaks and len(earlyPeaks) > 0 and len(latePeaks) > 0:
            # Rate Decay
            earlyDuration = (earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex']) * (1 / 60)
            lateDuration = (latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex']) * (1 / 60)

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
    cvCycleDuration = stdCycleDuration / meanCycleDuration if meanCycleDuration != 0 else np.nan
    cvRMSVelocity = stdRMSVelocity / meanRMSVelocity if meanRMSVelocity != 0 else np.nan
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed if meanAverageOpeningSpeed != 0 else np.nan
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed if meanAverageClosingSpeed != 0 else np.nan

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
        "MeanCycleDuration": meanCycleDuration,
        "StdCycleDuration": stdCycleDuration,
        "RangeCycleDuration": rangeCycleDuration,
        "Rate": rate,
        "AmplitudeDecay": amplitudeDecay,
        "VelocityDecay": velocityDecay,
        "RateDecay": rateDecay,
        "CVAmplitude": cvAmplitude,
        "CVCycleDuration": cvCycleDuration,
        "CVSpeed": cvSpeed,
        "CVRMSVelocity": cvRMSVelocity,
        "CVOpeningSpeed": cvAverageOpeningSpeed,
        "CVClosingSpeed": cvAverageClosingSpeed,
    }
    return jsonFinal

def get_fileName(file, outputFolder, scalingMethod):
    baseName = os.path.splitext(file)[0]
    # Ensure scalingMethod is suitable for filenames
    scalingMethodStr = scalingMethod.lower().replace(' ', '_')
    newFileName = f"{baseName}_{scalingMethodStr}.csv"
    return os.path.join(outputFolder, newFileName)

def main():
    inputFolder = 'data'
    outputFolder = 'output'  # Ensure this folder exists
    scalingMethod = 'THUMBSIZE'  # You can change this as needed
    listFiles = os.listdir(inputFolder)

    # Ensure the output directory exists
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for file in listFiles:
        print(file)
        if 'json' in file:
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
                # Compute scaling factor
                scalingFactor, actualScalingMethod = scaling(landMarks, scalingMethod)
                # Scale signal and recompute parameters
                outParameters = get_output(np.array(linePlotData) * scalingFactor)
                if outParameters is not None:
                    # Save to CSV
                    cvsFilename = get_fileName(file, outputFolder, actualScalingMethod)
                    pd.DataFrame.from_dict(data=outParameters, orient='index').to_csv(cvsFilename, header=False)
                else:
                    print(f"Skipping file {file} due to insufficient data.")

if __name__ == "__main__":
    main()