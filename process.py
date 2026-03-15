import os
import json
from turtle import speed
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

def ProcessCustomPeaks(distance,time, start_time,peaks,valleys_start,valleys_end,fs=60, cutOffFrequency=10): 

    b, a = signal.butter(2, cutOffFrequency, fs=fs, btype='low', analog=False)
    distance = signal.filtfilt(b, a, distance)  
    velocity = signal.savgol_filter(distance, 9, 3, deriv=1) / (1 / fs)

    informationPeaks = []
    for vs, ve, p in zip(valleys_start['time'], valleys_end['time'], peaks['time']):

        #adjust time 
        vs = vs - start_time
        ve = ve - start_time
        p = p - start_time
        #find the index of the peak and valleys
        openingValleyIndex = np.argmin(np.abs(time - vs))-1
        openingValleyIndex = openingValleyIndex if openingValleyIndex >= 0 else 0
        closingValleyIndex = np.argmin(np.abs(time - ve))+1
        closingValleyIndex = closingValleyIndex if closingValleyIndex<=len(distance) else len(distance)
        peakIndex = np.argmin(np.abs(time - p))


        #find the opening peak index by moving right from the opening valley until you find where the velocity is zero (end of positive segment)
        deriv = velocity.copy()
        deriv[deriv < 0] = 0
        deriv = deriv ** 2

        #get the max speed point between opening valley and peak
        if openingValleyIndex > 0 :
            openingMaxSpeedIndex = (openingValleyIndex-1) + np.argmax(deriv[openingValleyIndex-1:peakIndex])
        else:
            openingMaxSpeedIndex = (openingValleyIndex-1) + np.argmax(deriv[openingValleyIndex:peakIndex])

        #find the peak of the opening sequence
        idxPeak = openingMaxSpeedIndex + 1
        if idxPeak < len(deriv):
            while deriv[idxPeak] != 0:
                if idxPeak >= len(deriv) - 1:
                    idxPeak = np.nan
                    break

                idxPeak += 1
        openingPeakIndex = idxPeak

        #update the valley 
        idxValley = openingMaxSpeedIndex - 1
        if idxValley >= 0:
            while deriv[idxValley] != 0:
                if idxValley <= 0:
                    idxValley = np.nan
                    break

                idxValley -= 1
        openingValleyIndex = idxValley

        #find the closing peak index by moving left from the closing valley until you find where the velocity is zero (end of negative segment)
        deriv = velocity.copy()
        deriv[deriv > 0] = 0
        deriv = deriv ** 2


        if closingValleyIndex <= len(deriv):
            closingMaxSpeedIndex = (peakIndex) + np.argmax(deriv[peakIndex:closingValleyIndex+1])
        else:
            closingMaxSpeedIndex = (peakIndex) + np.argmax(deriv[peakIndex:closingValleyIndex])

        #find the peak of the closing sequence
        idxPeak = closingMaxSpeedIndex - 1
        if idxPeak >= 0:
            # Move left until you find where the velocity is zero (end of negative segment)
            while deriv[idxPeak] != 0:
                if idxPeak <= 0:
                    # If you reach the start of the signal, set idxPeak to NaN and break
                    idxPeak = np.nan
                    break
                idxPeak -= 1
        closingPeakIndex = idxPeak

        #update the valley of the closing sequence
        idxValley = closingMaxSpeedIndex + 1
        if idxValley < len(deriv):
            # Move right until you find where the velocity is zero (end of negative segment)
            while deriv[idxValley] != 0:
                if idxValley >= len(deriv) - 1:
                    # If you reach the end of the signal, set idxValley to NaN and break
                    idxValley = np.nan
                    break
                idxValley += 1
        closingValleyIndex = idxValley

        #put all this information together in a dictionary
        peakInfo = {
                'openingPeakIndex': openingPeakIndex ,
                'closingPeakIndex': closingPeakIndex ,
                'openingValleyIndex': openingValleyIndex,
                'openingMaxSpeedIndex': openingMaxSpeedIndex,
                'closingValleyIndex': closingValleyIndex,
                'closingMaxSpeedIndex': closingMaxSpeedIndex,
                'peakIndex': peakIndex,
                }
        
        if not (np.isnan(openingPeakIndex) or np.isnan(closingPeakIndex) or np.isnan(openingValleyIndex) or np.isnan(closingValleyIndex) or np.isnan(openingMaxSpeedIndex) or np.isnan(closingMaxSpeedIndex) or np.isnan(peakIndex)):
            informationPeaks.append(peakInfo)

    return distance, velocity, informationPeaks

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

def _theil_sen_slope(x, y):
    """
    Robust slope estimate: median of all pairwise slopes (Theil–Sen).
    Returns np.nan if <2 valid points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 2:
        return np.nan

    slopes = []
    for i in range(n - 1):
        dx = x[i+1:] - x[i]
        dy = y[i+1:] - y[i]
        valid = dx != 0
        if np.any(valid):
            slopes.extend((dy[valid] / dx[valid]).tolist())

    return float(np.nanmedian(slopes)) if len(slopes) else np.nan

def _robust_median(s): 
    return float(np.nanmedian(s.values)) if len(s)>= 2 else np.nan

def _robust_std(s):
    return float(np.nanstd(s, ddof=1)) if len(s) >= 2 else np.nan

def _cv(s):
    v = s.values.astype(float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan
    mu = np.mean(v)
    return float(np.std(v, ddof=1) / mu) if mu != 0 else np.nan

def compute_measures_per_cycle(peaks, distance, velocity, fs):
    """
    Docstring for compute_measures_per_cycle
    
    :param up_sample_signal: input signal

    Compute measures per cycle 
    """
    x = np.asarray(distance, dtype=float)
    v = np.asarray(velocity, dtype=float)
    maxVelocity = v.max()
    velocityThreshold = maxVelocity * 0.5

    rows = []

    for i, d in enumerate(peaks, start = 0):

        ov  = int(d["openingValleyIndex"])
        p   = int(d["peakIndex"])
        cv  = int(d["closingValleyIndex"])
        oms = int(d["openingMaxSpeedIndex"])
        cms = int(d["closingMaxSpeedIndex"])

        # Guard against bad ordering / indices
        if not (0 <= ov < len(x) and 0 <= p < len(x) and 0 <= cv < len(x)
                and 0 <= oms < len(x) and 0 <= cms < len(x)):
            continue
        if not (ov < p < cv):
            continue
        if not (ov <= oms <= p):
            # still compute but cap within segment
            oms = min(max(oms, ov), p)
        if not (p <= cms <= cv):
            cms = min(max(cms, p), cv)

        x_ov, x_p, x_cv = x[ov], x[p], x[cv]
        v_ov, v_p, v_cv = v[ov], v[p], v[cv]

        #amplitude, average opening valley to peak and closing valley to peak
        openingAmplitude = (x_p - x_ov)
        openingDuration = (p-ov) / fs
        closingAmplitude = (x_p - x_cv)
        closingDuration = (cv-p)/fs
        
        cycleAmplitude = (openingAmplitude + closingAmplitude)/2

        movementDuration = (cv-ov)/fs
        cycleSpeed = cycleAmplitude / movementDuration if movementDuration>0 else np.nan
        openingSpeed = openingAmplitude / openingDuration if openingDuration>0 else np.nan
        closingSpeed = closingAmplitude / closingDuration if closingDuration>0 else np.nan

        RMSvelocity = np.sqrt(np.mean(v[ov:cv] ** 2))

        openingSpeedMax = abs(v[oms])
        closingSpeedMax = abs(v[cms])

        #check is there are hesitations within the cycle 
        # calculate hesitations in the openeing-closing movement sequences 
        abs_velocity_segment = np.abs(v[ov+1:cv-1])
        # Count the number of times the absolute value of velocity crosses the threshold defined by maxVelocity*0.25
        crossingsDuringOpeningClosingMovement = np.sum((abs_velocity_segment[:-1] < velocityThreshold) & (abs_velocity_segment[1:] >= velocityThreshold)) + \
                    np.sum((abs_velocity_segment[:-1] >= velocityThreshold) & (abs_velocity_segment[1:] < velocityThreshold))
        

        cycleDuration = np.nan
        crossingsDuringPause = 0
        if i < (len(peaks) - 1): #don't do it for the last cycle
            ov_nextValley  = int(peaks[i+1]["openingValleyIndex"])
            cycleDuration = (ov_nextValley - ov) / fs #get the cycle duration, including from the opening vally of the current cycle until the opening valley of the following cycle

            #calculate hesitations in the pause between cycles
            pauseDuration = ((ov_nextValley) - cv) / fs
            if pauseDuration > 0.2 : #only consider pauses longer than 0.2 seconds 
                abs_velocity_segment = np.abs(v[cv:ov_nextValley-1])
                # Count the number of times the absolute value of velocity crosses the threshold defined by maxVelocity*velocityThreshold
                crossingsDuringPause = np.sum((abs_velocity_segment[:-1] < velocityThreshold) & (abs_velocity_segment[1:] >= velocityThreshold)) + \
                            np.sum((abs_velocity_segment[:-1] >= velocityThreshold) & (abs_velocity_segment[1:] < velocityThreshold))
                
        if crossingsDuringOpeningClosingMovement > 4 or  crossingsDuringPause > 4:
            hesitations = 1
        else:
            hesitations = 0

        rows.append({
            "cycle": i+1,
            "ov_idx": ov, "p_idx": p, "cv_idx": cv, "oms_idx": oms, "cms_idx": cms,
            "t_ov": ov / fs, "t_p": p / fs, "t_cv": cv / fs, "t_oms": oms / fs, "t_cms": cms / fs,
            "x_ov": x_ov, "x_p": x_p, "x_cv": x_cv,
            "v_ov": v_ov, "v_p": v_p, "v_cv": v_cv,
            "amplitude": cycleAmplitude,
            "duration": movementDuration,#cycleDuration,
            "speed": cycleSpeed,
            "RMSvelocity": RMSvelocity,
            "openingSpeed": openingSpeed,
            "closigSpeed": closingSpeed,
            "openingSpeedMax": openingSpeedMax,
            "closingSpeedMax": closingSpeedMax,
            "hesitations": hesitations,
            "totalDuration": cycleDuration,
        })

    return pd.DataFrame(rows)

def get_outputUpdated(distance, velocity = None, peaks = None, fs=None, desiredPeaks = 'all'):
   
    if peaks is None or velocity is None:
        fs = 60 if fs is None else fs
        distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(
            distance, fs=fs, minDistance=3, cutOffFrequency=7.5, prct=0.05
        )

    Npeaks = len(peaks)
    if desiredPeaks != 'all':
        #try to get a number from desiredPeaks and if it is a valid number and less than the number of peaks, use it to select the first desiredPeaks peaks
        try:    
            desiredPeaks = int(desiredPeaks)
            if desiredPeaks <= Npeaks+1:
                peaks = peaks[:desiredPeaks]
                Npeaks = len(peaks)
        except:
            print(f"desiredPeaks should be 'all' or an string representing an integer")


    cycles = compute_measures_per_cycle(peaks, distance, velocity, fs)

    feats = {}
    cycleDuration = cycles["duration"].dropna()
    
    # --- Slowness --- 
    feats["MedianSpeed"] =  _robust_median(cycles["speed"])
    feats["MedianRMSVelocity"] = _robust_median(cycles["RMSvelocity"])
    # feats["MedianClosingSpeed"] = _robust_median(cycles["closigSpeed"])
    # feats["MedianOpeningSpeed"] = _robust_median(cycles["openingSpeed"])
    feats["MedianMaxClosingSpeed"] = _robust_median(cycles["closingSpeedMax"])
    feats["MedianMaxOpeningSpeed"] = _robust_median(cycles["openingSpeedMax"])

    feats["MedianCycleDuration"] = _robust_median(cycleDuration)
    # feats["frequency"] = len(cycles["totalDuration"].dropna())/cycles["totalDuration"].sum() # (1.0 / feats["MedianCycleDuration"]) if np.isfinite(feats["MedianCycleDuration"]) and feats["MedianCycleDuration"] > 0 else np.nan
    feats["RangeCycleDuration"] = (cycleDuration.max() - cycleDuration.min()) if len(cycleDuration) >= 2 else np.nan
    
    # --- Hypokinesia ---
    feats["MedianAmplitude"] = _robust_median(cycles["amplitude"])


    # --- Sequence effect / decay  ---
    
    # idx = cycles["cycle"].values.astype(float)
    # feats["SlopeAmplitude"] = _theil_sen_slope(idx, cycles["amplitude"].values.astype(float))    
    # feats["SlopeSpeed"] = _theil_sen_slope(idx, cycles["speed"].values.astype(float))
    # # slope of cycle duration (cycle_time exists only for first N-1)
    # if len(cycleDuration) >= 2:
    #     idx_ct = cycles.loc[cycleDuration.index, "cycle"].values.astype(float)
    #     feats["SlopeCycleDuration"] = _theil_sen_slope(idx_ct, cycleDuration.values.astype(float))
    # else:
    #     feats["SlopeCycleDuration"] = np.nan

    amplitudeDecay = 1.0
    velocityDecay = 1.0
    timeDecay = 1.0

    # Check if there are enough peaks to compute decay parameters
    if len(cycles) >= 3:
        n = len(cycles) // 3 # Split the peaks into 3 parts
        if n == 0:
            n = 1  # Ensure at least one peak is selected

        earlyCycles = cycles.iloc[:n]
        lateCycles = cycles.iloc[-n:]

        # Ensure earlyPeaks and latePeaks are not empty
        if earlyCycles is not None and lateCycles is not None and len(earlyCycles) > 0 and len(lateCycles) > 0:
     
            # Amplitude Decay
            earlyAmplitude = _robust_median(earlyCycles["amplitude"])
            lateAmplitude = _robust_median(lateCycles["amplitude"])
            if np.mean(lateAmplitude) != 0:
                amplitudeDecay = earlyAmplitude / lateAmplitude
            else:
                amplitudeDecay = np.nan

            # Velocity Decay
            earlySpeed = _robust_median(earlyCycles["speed"])
            lateSpeed = _robust_median(lateCycles["speed"])
            if np.mean(lateSpeed) != 0:
                velocityDecay = earlySpeed / lateSpeed
            else:
                velocityDecay = np.nan

            #time decay 
            earlyCycleDuration = _robust_median(earlyCycles["duration"])
            lateCycleDuration = _robust_median(lateCycles["duration"])
            if np.mean(lateCycleDuration) != 0:
                timeDecay = earlyCycleDuration / lateCycleDuration
            else:
                timeDecay = np.nan

    feats["AmplitudeDecay"] = abs(1-amplitudeDecay)   
    feats["SpeedDecay"] = abs(1-velocityDecay) 
    feats["CycleDurationDecay"] = abs(1-timeDecay)     

    # --- Irregularity / hesitations / halts  ---
    feats["CVAmplitude"] = _cv(cycles["amplitude"])#_robust_std(cycles["amplitude"]) 
    feats["CVSpeed"] = _cv((cycles["speed"]))#_robust_std(cycles["speed"])
    feats["CVRMSVelocity"] = _cv(cycles["RMSvelocity"])
    feats["CVCycleDuration"] = _cv(cycleDuration)
    feats["CVMaxClosingSpeed"] = _cv(cycles["closingSpeedMax"])    
    feats["CVMaxOpeningSpeed"] = _cv(cycles["openingSpeedMax"])
    feats["NumberHesitations"] = int(cycles["hesitations"].sum())
    if len(cycleDuration) >= 2:
        feats["NumberPauses"] = int(((cycles["duration"] > 2 * feats["MedianCycleDuration"]).sum()) if np.isfinite(feats["MedianCycleDuration"]) else 0)
    else:
        feats["NumberPauses"] = 0

    return feats, distance, velocity

def get_output(distance, velocity = None, peaks = None, fs=None, desiredPeaks = 'all'):

    if peaks is None or velocity is None:
        fs = 60 if fs is None else fs
        distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(
            distance, fs=fs, minDistance=3, cutOffFrequency=7.5, prct=0.05
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

    #check if the subjects wans to use certain number of peaks 
    Npeaks = len(peaks)
    if desiredPeaks != 'all':
        #try to get a number from desiredPeaks and if it is a valid number and less than the number of peaks, use it to select the first desiredPeaks peaks
        try:    
            desiredPeaks = int(desiredPeaks)
            if desiredPeaks <= Npeaks+1:
                peaks = peaks[:desiredPeaks]
                Npeaks = len(peaks)
        except:
            print(f"desiredPeaks should be 'all' or an string representing an integer")


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
    parser.add_argument("--scalingMethod", type=str, required=False, default='NOSCALING', choices=['THUMBSIZE', 'INDEXSIZE', 'PALMSIZE', 'NOSCALING'], help="Scaling method to use.")
    parser.add_argument("--estimatePeaks", type=bool, required=False, default=False, choices=[True,False], help="Estimate peaks if not provided in the JSON file.")
    parser.add_argument("--desiredPeaks", type=str, required=False, default="all",  help="Number of peaks to use during estimation")
    parser.add_argument("--plotResults", type=bool, required=False, default=True, choices=[True,False],  help="Plot resulting signal")
    parser.add_argument("--featuresEstimator", type=str, required=False, default="default", choices=["default", "new"], help="Feature estimator to use")
    args = parser.parse_args()

    inputFolder = args.inputFolder
    outputFolder = args.outputFolder
    scalingMethod = args.scalingMethod
    estimatePeaks = args.estimatePeaks
    desiredPeaks = args.desiredPeaks
    plotResults = args.plotResults
    featuresEstimator = args.featuresEstimator
    if featuresEstimator == 'default':
        estimator = get_output
    else:
        estimator = get_outputUpdated

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
                        # outParameters, distance, velocity = get_outputUpdated(up_sample_signal)
                        if estimatePeaks == True:
                            outParameters, distance, velocity = estimator(up_sample_signal, desiredPeaks = desiredPeaks)
                        else:
                            distance, velocity, peaks = ProcessCustomPeaks(up_sample_signal,time_vector, start_time, data['peaks'], data['valleys_start'], data['valleys_end'], fs=60, cutOffFrequency=7.5)
                            outParameters, distance, velocity = estimator(distance, velocity, peaks, fs=60, desiredPeaks = desiredPeaks)

                        actualScalingMethod = 'NOSCALING'
                    else:
                        scalingFactor, actualScalingMethod = scaling(landMarks, scalingMethod)
                        # Scale signal and recompute parameters
                        # outParameters, distance, velocity = get_outputUpdated(up_sample_signal * scalingFactor)
                        if estimatePeaks == True:
                            outParameters, distance, velocity = estimator(up_sample_signal * scalingFactor, desiredPeaks = desiredPeaks)
                        else:
                            distance, velocity, peaks = ProcessCustomPeaks(up_sample_signal * scalingFactor,time_vector, start_time, data['peaks'], data['valleys_start'], data['valleys_end'], fs=60, cutOffFrequency=7.5)
                            outParameters, distance, velocity = estimator(distance , velocity, peaks, fs=60, desiredPeaks = desiredPeaks)


       
                    if outParameters is not None:
                        # Save to CSV
                        cvsFilename = get_fileName(file, outputFolder, actualScalingMethod)
                        pd.DataFrame.from_dict(data=outParameters, orient='index').to_csv(cvsFilename, header=False)
                        if plotResults:
                            plt.figure(figsize=(10, 6))
                            plt.plot(np.arange(len(distance)) / 60, np.array(distance))
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
                            plt.plot(linePlotData, linePlotData)
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