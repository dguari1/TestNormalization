import numpy as np
import scipy.signal as signal


def compareNeighboursNegative(item1, item2, distance, minDistance=5):
    """
    This function compares two neighboring items (item1 and item2) in a signal and determines whether they are too close
    based on their peak and valley indices. If they are too close, it decides which one to keep based on their maximum
    speed. The function handles four cases:
    1. item1's valley and item2's peak are too close.
    2. item1's peak and item2's peak are too close.
    3. item1's valley and item2's valley are too close.
    4. item1's valley and item2's peak have similar heights.

    In each case, the function returns a new item with the appropriate peak, valley, and max speed values. If none of
    the cases apply, it returns None.
    """
    # case 1 -> item1 peak and item2 valley are too close
    if abs(item1['valleyIndex'] - item2['peakIndex']) < minDistance:
        # Merge the two items to create a new item
        # this merge is based on the item with max speed

        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {}
            newItem['maxSpeedIndex'] = item1['maxSpeedIndex']
            newItem['maxSpeed'] = item1['maxSpeed']
            newItem['peakIndex'] = item1['peakIndex']
            newItem['valleyIndex'] = item2['valleyIndex']
        else:
            newItem = {}
            newItem['maxSpeedIndex'] = item2['maxSpeedIndex']
            newItem['maxSpeed'] = item2['maxSpeed']
            newItem['peakIndex'] = item1['peakIndex']
            newItem['valleyIndex'] = item2['valleyIndex']

        return newItem

    # case 2 -> item1 peak and item2 peak are too close
    if abs(item1['peakIndex'] - item2['peakIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = item1
        else:
            newItem = item2

        return newItem

    # case 3 -> item1 valley and item2 valley are too close
    if abs(item1['valleyIndex'] - item2['valleyIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = item1
        else:
            newItem = item2
        # skip item2
        return newItem

    # case 4-> item1 valley is of similar height to item2 peak
    if abs(distance[item1['valleyIndex']] - distance[item2['peakIndex']]) < abs(
            distance[item1['valleyIndex']] - distance[item1['maxSpeedIndex']]) / 5:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {}
            newItem['maxSpeedIndex'] = item1['maxSpeedIndex']
            newItem['maxSpeed'] = item1['maxSpeed']
            newItem['peakIndex'] = item1['peakIndex']
            newItem['valleyIndex'] = item2['valleyIndex']
        else:
            newItem = {}
            newItem['maxSpeedIndex'] = item2['maxSpeedIndex']
            newItem['maxSpeed'] = item2['maxSpeed']
            newItem['peakIndex'] = item1['peakIndex']
            newItem['valleyIndex'] = item2['valleyIndex']

        return newItem

    return None

def compareNeighboursPositive(item1, item2, distance, minDistance=5):
    """
    This function compares two neighboring items (item1 and item2) in a signal and determines whether they are too close
    based on their peak and valley indices. If they are too close, it decides which one to keep based on their maximum
    speed. The function handles four cases:
    1. item1's valley and item2's peak are too close.
    2. item1's peak and item2's peak are too close.
    3. item1's valley and item2's valley are too close.
    4. item1's valley and item2's peak have similar heights.

    In each case, the function returns a new item with the appropriate peak, valley, and max speed values. If none of
    the cases apply, it returns None.
    """
    # case 1 -> item1 peak and item2 valley are too close
    if abs(item1['peakIndex'] - item2['valleyIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {}
            newItem['maxSpeedIndex'] = item1['maxSpeedIndex']
            newItem['maxSpeed'] = item1['maxSpeed']
            newItem['peakIndex'] = item2['peakIndex']
            newItem['valleyIndex'] = item1['valleyIndex']
        else:
            newItem = {}
            newItem['maxSpeedIndex'] = item2['maxSpeedIndex']
            newItem['maxSpeed'] = item2['maxSpeed']
            newItem['peakIndex'] = item2['peakIndex']
            newItem['valleyIndex'] = item1['valleyIndex']

        return newItem

    # case 2 -> item1 peak and item2 peak are too close
    if abs(item1['peakIndex'] - item2['peakIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = item1
        else:
            newItem = item2

        return newItem

    # case 3 -> item1 valley and item2 valley are too close
    if abs(item1['valleyIndex'] - item2['valleyIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = item1
        else:
            newItem = item2

        return newItem

    # case 4-> item1 valley is of similar height to item2 peak
    if abs(distance[item1['peakIndex']] - distance[item2['valleyIndex']]) < abs(
            distance[item1['peakIndex']] - distance[item1['maxSpeedIndex']]) / 5:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {}
            newItem['maxSpeedIndex'] = item1['maxSpeedIndex']
            newItem['maxSpeed'] = item1['maxSpeed']
            newItem['peakIndex'] = item2['peakIndex']
            newItem['valleyIndex'] = item1['valleyIndex']
        else:
            newItem = {}
            newItem['maxSpeedIndex'] = item2['maxSpeedIndex']
            newItem['maxSpeed'] = item2['maxSpeed']
            newItem['peakIndex'] = item2['peakIndex']
            newItem['valleyIndex'] = item1['valleyIndex']

        return newItem

    return None

def eliminateBadNeighboursNegative(indexVelocity, distance, minDistance=5):
    """
    Eliminates "bad neighbors" from a list of indices based on a comparison of their distances.
    This function processes a list of indices (`indexVelocity`) and removes or adjusts neighboring
    indices that are too close to each other based on a specified minimum distance (`minDistance`).
    It uses a helper function `compareNeighboursNegative` to determine whether to keep, replace, or
    skip neighboring indices. The function ensures that no two indices in the returned list are
    closer than the specified minimum distance.
    Args:
        indexVelocity (list of int): A list of indices representing positions or velocities.
        distance (list of float): A list of distances corresponding to the indices in `indexVelocity`.
                                   This is used to evaluate the proximity of neighboring indices.
        minDistance (int, optional): The minimum allowable distance between neighboring indices.
                                     Defaults to 5.
    Returns:
        list of int: A corrected list of indices where "bad neighbors" have been eliminated or adjusted.
    Notes:
        - The function uses a boolean list `isSkip` to track which indices should be skipped during
          processing.
        - The helper function `compareNeighboursNegative` is expected to return either a new index
          to replace a pair of neighbors or `None` if no replacement is needed.
        - If an index is skipped, it will not be included in the final corrected list.
    Example:
        Suppose `indexVelocity = [1, 3, 8, 10]`, `distance = [2.0, 1.5, 0.5, 3.0]`, and `minDistance = 5`.
        The function will process the indices and return a corrected list based on the logic in
        `compareNeighboursNegative`.
    Dependencies:
        This function relies on the external helper function `compareNeighboursNegative`, which must
        be defined elsewhere in the codebase. The behavior of this function directly affects the
        output of `eliminateBadNeighboursNegative`.
    """

    indexVelocityCorrected = []
    isSkip = [False] * len(indexVelocity)

    for idx in range(len(indexVelocity)):

        if isSkip[idx] == False:  # do not skip this item

            if idx < len(indexVelocity) - 1:

                newItem = compareNeighboursNegative(indexVelocity[idx], indexVelocity[idx + 1], distance, minDistance)
                if newItem is not None:
                    # newItem was returned, save returned element and skip following element
                    indexVelocityCorrected.append(newItem)
                    isSkip[idx + 1] = True
                else:
                    # no new Item, keep current item
                    indexVelocityCorrected.append(indexVelocity[idx])
            else:
                indexVelocityCorrected.append(indexVelocity[idx])

    return indexVelocityCorrected

def eliminateBadNeighboursPositive(indexVelocity, distance, minDistance=5):
    """
    Eliminates "bad neighbors" in a list of indices based on a distance criterion.
    This function processes a list of indices (`indexVelocity`) and removes or adjusts
    neighboring indices that are too close to each other based on the provided `distance`
    and `minDistance` parameters. The function ensures that the resulting list of indices
    (`indexVelocityCorrected`) adheres to the specified minimum distance constraint.
    Args:
        indexVelocity (list of int): A list of indices representing positions or velocities.
        distance (list of float): A list of distances corresponding to the indices in `indexVelocity`.
                                   It is assumed that `distance` has the same length as `indexVelocity`.
        minDistance (int, optional): The minimum allowable distance between neighboring indices.
                                     Defaults to 5.
    Returns:
        list of int: A corrected list of indices (`indexVelocityCorrected`) where "bad neighbors"
                     have been eliminated or adjusted based on the distance criteria.
    Notes:
        - The function uses a helper function `compareNeighboursPositive` (not provided here)
          to determine whether neighboring indices should be adjusted or kept.
        - The `isSkip` list is used to track which indices should be skipped during processing
          to avoid duplicate or unnecessary comparisons.
        - If the current index is the last in the list, it is automatically added to the
          corrected list without further checks.
    Example:
        Given the following inputs:
            indexVelocity = [1, 3, 8, 12]
            distance = [2.0, 4.5, 1.2, 3.8]
            minDistance = 5
        The function will process the indices and return a corrected list of indices
        based on the distance criteria.
    """

    indexVelocityCorrected = []
    isSkip = [False] * len(indexVelocity)

    for idx in range(len(indexVelocity)):

        if isSkip[idx] == False:  # do not skip this item

            if idx < len(indexVelocity) - 1:

                newItem = compareNeighboursPositive(indexVelocity[idx], indexVelocity[idx + 1], distance,
                                                    minDistance=minDistance)
                if newItem is not None:
                    # newItem was returned, save returned element and skip following element
                    indexVelocityCorrected.append(newItem)
                    isSkip[idx + 1] = True
                else:
                    # no new Item, keep current item
                    indexVelocityCorrected.append(indexVelocity[idx])
            else:
                indexVelocityCorrected.append(indexVelocity[idx])

    return indexVelocityCorrected

def correctBasedonHeight(pos, distance, prct=0.125, minDistance=5):
    """
    This function is deprecated and no longer used.
    Filters a list of peak and valley positions based on their height relative to the average height 
    of all peaks and valleys, as well as additional constraints such as minimum distance and 
    positional relationships.
    Parameters:
    -----------
    pos : list of dict
        A list of dictionaries where each dictionary represents a peak and valley pair. Each dictionary 
        is expected to contain the following keys:
        - 'peakIndex': Index of the peak in the `distance` array.
        - 'valleyIndex': Index of the valley in the `distance` array.
        - 'maxSpeedIndex': Index of the maximum speed in the `distance` array.
    distance : list or numpy array
        A sequence of numerical values representing the signal from which peaks and valleys are derived.
    prct : float, optional (default=0.125)
        The percentage of the average peak height used as a threshold. Peaks with heights below this 
        threshold are discarded.
    minDistance : int, optional (default=5)
        The minimum allowable distance between a peak and its corresponding valley. Pairs with distances 
        below this threshold are discarded.
    Returns:
    --------
    corrected : list of dict
        A filtered list of dictionaries containing only the peak and valley pairs that meet the specified 
        criteria.
    Notes:
    ------
    - The function calculates the height of each peak as the absolute difference between the `distance` 
      values at the peak and valley indices.
    - A peak is retained if:
        1. Its height is greater than `prct` times the average peak height.
        2. The distance between its peak and valley indices is greater than or equal to `minDistance`.
        3. The peak value is greater than the value at `maxSpeedIndex`, and the valley value is less than 
           the value at `maxSpeedIndex`.
    - Any exceptions encountered during processing are silently ignored.
    - This function is no longer in use and may be removed in future versions.
    """

    # eliminate any peaks that is smaller than 15% of the average height
    heightPeaks = []
    for item in pos:
        try:
            heightPeaks.append(abs(distance[item['peakIndex']] - distance[item['valleyIndex']]))
        except:
            pass

    meanHeightPeak = np.mean(heightPeaks)
    corrected = []
    for item in pos:
        try:
            if (abs(distance[item['peakIndex']] - distance[item['valleyIndex']])) > prct * meanHeightPeak:
                if abs(item['peakIndex'] - item['valleyIndex']) >= minDistance:
                    if (distance[item['peakIndex']] > distance[item['maxSpeedIndex']]) and (
                            distance[item['valleyIndex']] < distance[item['maxSpeedIndex']]):
                        corrected.append(item)
                    else:
                        pass
                else:
                    pass
            else:
                pass
        except:
            pass

    return corrected

def correctBasedonVelocityNegative(pos, velocity, prct=0.125):
    """
    This function is deprecated and no longer in use.
    Filters a list of position dictionaries based on their associated velocity values.
    The function removes positive velocities, calculates the mean of the squared velocity peaks, and retains only
    the positions where the velocity at the 'maxSpeedIndex' exceeds a specified 
    percentage of the mean velocity peaks.
    Args:
        pos (list of dict): A list of dictionaries, each containing a 'maxSpeedIndex' key
                            that corresponds to an index in the velocity array.
        velocity (numpy.ndarray): A 1D array of velocity values. Negative values are set to 0,
                                  and all values are squared before processing.
        prct (float, optional): A percentage threshold (default is 0.125 or 12.5%) used to 
                                filter positions based on their velocity.
    Returns:
        list of dict: A filtered list of position dictionaries where the velocity at the 
                      'maxSpeedIndex' exceeds the specified percentage of the mean velocity peaks.
    Note:
        - This function modifies the input velocity array by setting positive values to 0.
        - It is recommended to use an updated or alternative function for similar functionality.
    """
    velocity[velocity>0] = 0
    velocity = velocity ** 2

    velocityPeaks = []
    for item in pos:
        try:
            velocityPeaks.append(velocity[item['maxSpeedIndex']])
        except:
            pass

    meanvelocityPeaks = np.mean(velocityPeaks)

    corrected = []
    for item in pos:
        try:
            if (velocity[item['maxSpeedIndex']]) > prct * meanvelocityPeaks:
                corrected.append(item)
            else:
                pass
        except:
            pass

    return corrected

def correctBasedonVelocityPositive(pos, velocity, prct=0.125):
    """
    This function is deprecated and no longer in use.
    Filters a list of position dictionaries based on their associated velocity values.
    The function removes negative velocities calculates the mean of the squared velocity peaks and retains only
    the positions where the velocity at the 'maxSpeedIndex' exceeds a specified 
    percentage of the mean velocity peaks.
    Args:
        pos (list of dict): A list of dictionaries, each containing a 'maxSpeedIndex' key
                            that corresponds to an index in the velocity array.
        velocity (numpy.ndarray): A 1D array of velocity values. Negative values are set to 0,
                                  and all values are squared before processing.
        prct (float, optional): A percentage threshold (default is 0.125 or 12.5%) used to 
                                filter positions based on their velocity.
    Returns:
        list of dict: A filtered list of position dictionaries where the velocity at the 
                      'maxSpeedIndex' exceeds the specified percentage of the mean velocity peaks.
    Note:
        - This function modifies the input velocity array by setting negative values to 0.
        - It is recommended to use an updated or alternative function for similar functionality.
    """

    velocity[velocity < 0] = 0
    velocity = velocity ** 2

    velocityPeaks = []
    for item in pos:
        try:
            velocityPeaks.append(velocity[item['maxSpeedIndex']])
        except:
            pass

    meanvelocityPeaks = np.mean(velocityPeaks)

    corrected = []
    for item in pos:
        try:
            if (velocity[item['maxSpeedIndex']]) > prct * meanvelocityPeaks:
                corrected.append(item)
            else:
                pass
        except:
            pass

    return corrected

def correctFullPeaks(distance, pos, neg):
    """
    Corrects and identifies peak candidates in a signal based on opening and closing velocity indices.

    This function processes a signal to identify and correct peak candidates by analyzing the 
    relationships between opening and closing velocity indices. It ensures that overlapping or 
    duplicate peaks are resolved and returns a list of corrected peak candidates.

    Args:
        distance (numpy.ndarray): A 1D array representing the signal values.
        pos (list of dict): A list of dictionaries representing the positive (opening) velocity 
            information. Each dictionary must contain the keys:
                - 'maxSpeedIndex': Index of the maximum speed in the positive direction.
                - 'valleyIndex': Index of the valley in the positive direction.
                - 'peakIndex': Index of the peak in the positive direction.
        neg (list of dict): A list of dictionaries representing the negative (closing) velocity 
            information. Each dictionary must contain the keys:
                - 'maxSpeedIndex': Index of the maximum speed in the negative direction.
                - 'valleyIndex': Index of the valley in the negative direction.
                - 'peakIndex': Index of the peak in the negative direction.

    Returns:
        list of dict: A list of corrected peak candidates. Each dictionary in the list contains:
            - 'openingValleyIndex': Index of the valley in the opening phase.
            - 'openingPeakIndex': Index of the peak in the opening phase.
            - 'openingMaxSpeedIndex': Index of the maximum speed in the opening phase.
            - 'closingValleyIndex': Index of the valley in the closing phase.
            - 'closingPeakIndex': Index of the peak in the closing phase.
            - 'closingMaxSpeedIndex': Index of the maximum speed in the closing phase.
            - 'peakIndex': Index of the absolute peak within the identified range.

    Notes:
        - The function calculates the absolute peak within the range defined by the opening and 
          closing velocity indices.
        - If duplicate peaks are detected (i.e., peaks with the same index), they are merged into 
          a single corrected peak candidate.
        - The function uses numpy for numerical operations and assumes that the input arrays and 
          lists are properly formatted.

    Raises:
        None: The function uses a try-except block to handle potential errors during processing 
        and skips invalid cases silently.
    """
   
    closingVelocities = []
    for item in neg:
        closingVelocities.append(item['maxSpeedIndex'])

    openingVelocities = []
    for item in pos:
        openingVelocities.append(item['maxSpeedIndex'])

    peakCandidates = []
    for idx, closingVelocity in enumerate(closingVelocities):
        try:
            # Calculate the difference between the current closing velocity and all opening velocities
            difference = np.array(openingVelocities) - closingVelocity
            # Keep only negative differences (closing velocity must occur after opening velocity)
            difference[difference > 0] = 0

            # Find the index of the most negative difference (closest valid opening velocity)
            posmin = np.argmax(difference[np.nonzero(difference)])

            # Find the absolute peak value and its index within the range of the selected opening and closing velocities
            absolutePeak = np.max(distance[pos[posmin]['maxSpeedIndex']: neg[idx]['maxSpeedIndex'] + 1])
            absolutePeakIndex = pos[posmin]['maxSpeedIndex'] + np.argmax(
                distance[pos[posmin]['maxSpeedIndex']: neg[idx]['maxSpeedIndex'] + 1])

            # Create a dictionary to store the peak candidate's details
            peakCandidate = {}

            #preassign openingPeakIndex and closingPeakIndex to avoid error
            peakCandidate['openingPeakIndex'] = absolutePeakIndex
            peakCandidate['closingPeakIndex'] = absolutePeakIndex

            # Add details of the opening phase (valley, peak, and max speed indices)
            peakCandidate['openingValleyIndex'] = pos[posmin]['valleyIndex']
            peakCandidate['openingPeakIndex'] = pos[posmin]['peakIndex']
            peakCandidate['openingMaxSpeedIndex'] = pos[posmin]['maxSpeedIndex']

            # Add details of the closing phase (valley, peak, and max speed indices)
            peakCandidate['closingValleyIndex'] = neg[idx]['valleyIndex']
            peakCandidate['closingPeakIndex'] = neg[idx]['peakIndex']
            peakCandidate['closingMaxSpeedIndex'] = neg[idx]['maxSpeedIndex']

            # Add the index of the absolute peak within the range
            peakCandidate['peakIndex'] = absolutePeakIndex

            # Append the peak candidate to the list
            peakCandidates.append(peakCandidate)
        except:
            # Skip any errors that occur during processing
            pass

    peakCandidatesCorrected = []
    idx = 0
    while idx < len(peakCandidates):
        # Get the current peak candidate
        peakCandidate = peakCandidates[idx]
        peak = peakCandidate['peakIndex']

        # Calculate the difference between the index of the current peak and all other peaks
        difference = [(peak - item['peakIndex']) for item in peakCandidates]

        # If the current peak is unique (no duplicates), add it to the corrected list
        if len(np.where(np.array(difference) == 0)[0]) == 1:
            peakCandidatesCorrected.append(peakCandidate)
            idx += 1
        else:
            # If there are duplicate peaks, merge the two items into one
            item1 = peakCandidates[np.where(np.array(difference) == 0)[0][0]]
            item2 = peakCandidates[np.where(np.array(difference) == 0)[0][1]]

            # Create a new peak candidate by combining the opening information from item1
            # and the closing information from item2
            peakCandidate = {}

            #preassign openingPeakIndex and closingPeakIndex to avoid error
            peakCandidate['openingPeakIndex'] = absolutePeakIndex
            peakCandidate['closingPeakIndex'] = absolutePeakIndex

            peakCandidate['openingValleyIndex'] = item1['openingValleyIndex']
            peakCandidate['openingPeakIndex'] = item1['openingPeakIndex']
            peakCandidate['openingMaxSpeedIndex'] = item1['openingMaxSpeedIndex']

            peakCandidate['closingValleyIndex'] = item2['closingValleyIndex']
            peakCandidate['closingPeakIndex'] = item2['closingPeakIndex']
            peakCandidate['closingMaxSpeedIndex'] = item2['closingMaxSpeedIndex']

            # Use the peak index from item2 as the final peak index
            peakCandidate['peakIndex'] = item2['peakIndex']

            # Add the merged peak candidate to the corrected list
            peakCandidatesCorrected.append(peakCandidate)

            # Skip the next item since it has already been merged
            idx += 2

    return peakCandidatesCorrected

def correctBasedonPeakSymmetry(peaks):
    """
    Filters a list of peaks based on their symmetry.
    This function evaluates the symmetry of each peak in the input list by calculating
    the ratio of the distances between the peak's center and its left and right valleys.
    Peaks with a symmetry ratio within the range [0.25, 4] are considered valid and are
    included in the output list.
    Parameters:
    -----------
    peaks : list of dict
        A list of dictionaries where each dictionary represents a peak. Each dictionary
        must contain the following keys:
        - 'openingValleyIndex' (int): The index of the left valley of the peak.
        - 'peakIndex' (int): The index of the peak's center.
        - 'closingValleyIndex' (int): The index of the right valley of the peak.
    Returns:
    --------
    list of dict
        A list of dictionaries representing the peaks that satisfy the symmetry condition.
    Notes:
    ------
    - The symmetry ratio is calculated as:
    - A peak is considered symmetric if its ratio is within the range [0.25, 4].
    - Peaks that do not meet this condition are excluded from the output.
    Example:
    --------
    peaks = [
        {'openingValleyIndex': 2, 'peakIndex': 5, 'closingValleyIndex': 8},
        {'openingValleyIndex': 1, 'peakIndex': 4, 'closingValleyIndex': 10}
    ]
    result = correctBasedonPeakSymmetry(peaks)
    # result will contain only the peaks that satisfy the symmetry condition.
    """

    peaksCorrected = []
    for peak in peaks:
        leftValley = peak['openingValleyIndex']
        centerPeak = peak['peakIndex']
        rightValley = peak['closingValleyIndex']

        ratio = (centerPeak - leftValley) / (rightValley - centerPeak)
        if 0.25 <= ratio <= 4:
            peaksCorrected.append(peak)

    return peaksCorrected

def correctBasedonHeightSymmetry(peaks, distance):
    """
    Filters a list of peaks based on the symmetry of their height relative to their surrounding valleys.
    This function evaluates the symmetry of each peak by calculating the ratio of the distance 
    between the peak and its left valley to the distance between the peak and its right valley. 
    Peaks with a ratio within the range [0.25, 4] are considered symmetric and are included in 
    the output list.
    Args:
        peaks (list of dict): A list of dictionaries where each dictionary represents a peak. 
            Each dictionary must contain the following keys:
                - 'openingValleyIndex' (int): The index of the left valley in the `distance` list.
                - 'peakIndex' (int): The index of the peak in the `distance` list.
                - 'closingValleyIndex' (int): The index of the right valley in the `distance` list.
        distance (list of float): A list of numerical values representing the distances or heights 
            at each index. This is used to calculate the symmetry ratio for each peak.
    Returns:
        list of dict: A filtered list of peaks where each peak satisfies the symmetry condition 
        (ratio between 0.25 and 4). Peaks that do not meet this condition are excluded.
    Example:
        peaks = [
            {'openingValleyIndex': 0, 'peakIndex': 2, 'closingValleyIndex': 4},
            {'openingValleyIndex': 5, 'peakIndex': 7, 'closingValleyIndex': 9}
        ]
        distance = [1, 2, 5, 3, 1, 2, 4, 6, 3, 1]
        result = correctBasedonHeightSymmetry(peaks, distance)
        # result will contain only the peaks that satisfy the symmetry condition.
    """

    peaksCorrected = []
    for peak in peaks:
        leftValley = peak['openingValleyIndex']
        centerPeak = peak['peakIndex']
        rightValley = peak['closingValleyIndex']

        ratio = (abs(distance[centerPeak] - distance[leftValley])) / (abs(distance[rightValley] - distance[centerPeak]))
        if 0.25 <= ratio <= 4:
            peaksCorrected.append(peak)

    return peaksCorrected

def correctHeadsandTails(peaks):
    """
    Corrects the valleys of consecutive peaks if the ending valley of the first peak 
    occurs after the opening valley of the following peak. Swaps the positions of the 
    ending and opening valleys in such cases.

    Args:
        peaks (list of dict): A list of dictionaries where each dictionary represents a peak. 
            Each dictionary must contain the following keys:
                - 'openingValleyIndex' (int): The index of the opening valley.
                - 'closingValleyIndex' (int): The index of the closing valley.

    Returns:
        list of dict: A list of corrected peaks with swapped valleys where necessary.
    """
    for i in range(len(peaks) - 1):
        if peaks[i]['closingValleyIndex'] > peaks[i + 1]['openingValleyIndex']:
            # Swap the valleys
            peaks[i]['closingValleyIndex'] = peaks[i + 1]['openingValleyIndex']
            peaks[i + 1]['openingValleyIndex'] = peaks[i]['closingValleyIndex']
    return peaks

def correctBasedonHeightSymmetryRatio(peak, distance):
    """
    Calculates the height ratio of a peak based on its left and right valleys.
    This function computes the absolute height ratio of a peak by comparing the 
    distances to its left and right valleys. The ratio is defined as the absolute 
    difference between the peak and the left valley divided by the absolute difference 
    between the peak and the right valley.

    The funcion merges two consecutive peaks if both have a peakheightratio smaller than 0.5 or larger than 2.
    Peaks with a ratio within the range [0.5, 2] are considered symmetric and are included in
    the output list.

    The ratio is calculated as:
    ratio = (peak - leftValley) / (rightValley - peak)
    where:
        - peak: The index of the peak in the `distance` list.
        - leftValley: The index of the left valley in the `distance` list.
        - rightValley: The index of the right valley in the `distance` list.

    Args:
        peak (dict): A dictionary representing a peak. It must contain the following keys:
            - 'openingValleyIndex' (int): The index of the left valley in the `distance` list.
            - 'peakIndex' (int): The index of the peak in the `distance` list.
            - 'closingValleyIndex' (int): The index of the right valley in the `distance` list.
        distance (list of float): A list of numerical values representing distances or heights 
            at each index. This is used to calculate the height ratio for the peak.

    Returns:
        float: The absolute height ratio of the peak based on its left and right valleys.
    """
    peaksCorrected = []  # List to store the corrected peaks
    idx = 0  # Initialize the index for iterating through the peaks

    while idx < len(peak) - 1:  # Loop through the peaks until the second-to-last peak
        currentPeak = peak[idx]  # Get the current peak
        mergedPeak = currentPeak  # Initialize the merged peak as the current peak
        mergeOccurred = False  # Flag to track if a merge has occurred

        while idx < len(peak) - 1:  # Inner loop to check for merging with the next peak
            nextPeak = peak[idx + 1]  # Get the next peak
            # Calculate the height symmetry ratio for the current peak
            currentRatio = abs(distance[mergedPeak['peakIndex']] - distance[mergedPeak['openingValleyIndex']]) / abs(distance[mergedPeak['peakIndex']] - distance[mergedPeak['closingValleyIndex']])
            # Calculate the height symmetry ratio for the next peak
            nextRatio = abs(distance[nextPeak['peakIndex']] - distance[nextPeak['openingValleyIndex']]) / abs(distance[nextPeak['peakIndex']] - distance[nextPeak['closingValleyIndex']])

            # Check if both the current and next peaks have ratios outside the acceptable range
            if (currentRatio < 0.5 or currentRatio > 2) and (nextRatio < 0.5 or nextRatio > 2):
                # Merge the peaks by combining their properties
                mergedPeak['openingValleyIndex'] = mergedPeak['openingValleyIndex']  # Keep the opening valley of the first peak
                mergedPeak['openingMaxSpeedIndex'] = mergedPeak['openingMaxSpeedIndex'] # Keep the opening max speed index of the first peak
                # Choose the peak with the higher distance value as the merged peak
                mergedPeak['peakIndex'] = mergedPeak['peakIndex'] if distance[mergedPeak['peakIndex']] > distance[nextPeak['peakIndex']] else nextPeak['peakIndex']
                mergedPeak['closingValleyIndex'] = nextPeak['closingValleyIndex']  # Use the closing valley of the next peak
                mergedPeak['closingMaxSpeedIndex'] = nextPeak['closingMaxSpeedIndex']  # Use the closing max speed index of the next peak
                # mergedPeak = {
                #     'openingValleyIndex': mergedPeak['openingValleyIndex'],  # Keep the opening valley of the first peak
                #     'openingMaxSpeedIndex': mergedPeak['openingMaxSpeedIndex'],  # Keep the opening max speed index of the first peak
                #     # Choose the peak with the higher distance value as the merged peak
                #     'peakIndex': mergedPeak['peakIndex'] if distance[mergedPeak['peakIndex']] > distance[nextPeak['peakIndex']] else nextPeak['peakIndex'],
                #     'closingValleyIndex': nextPeak['closingValleyIndex'],  # Use the closing valley of the next peak
                #     'closingMaxSpeedIndex': nextPeak['closingMaxSpeedIndex']  # Use the closing max speed index of the next peak
                # }
                mergeOccurred = True  # Set the merge flag to True
                idx += 1  # Move to the next peak for further merging
            else:
                break  # Exit the inner loop if no merge condition is met

        # Add the merged peak to the corrected list if a merge occurred, otherwise add the current peak
        peaksCorrected.append(mergedPeak if mergeOccurred else currentPeak)
        idx += 1  # Move to the next peak

    peaksCorrected.append(peak[-1])  # Append the last peak to the corrected list

    return peaksCorrected  # Return the list of corrected peaks

def correctBasedonHeightandVelocityPositivePeaks(pos,distance,velocity, minDistance=5,prct=0.1):
    """Correct peaks based on height and velocity
    Args:
        pos (list): list of peaks
        distance (list): distance signal
        velocity (list): velocity signal
        minDistance (int, optional): minimum distance between peaks. Defaults to 5.
        prct (float, optional): percentage of the mean height. Defaults to 0.1.
    Returns:
        list: list of corrected peaks


    This function corrects the list of positive peaks based on certain conditions:
    1. Peaks with velocity and height smaller than a percentage (prct) of the maximum velocity and height are ignored.
    2. Peaks that are too close to their corresponding valleys (less than minDistance) are ignored.
    3. Peaks that occur after the maximum speed point or where the maximum speed point occurs after the valley are ignored.
    Peaks that satisfy all the conditions are added to the corrected list and returned.
    """
    #remove negative velocity and square it
    velocity[velocity < 0] = 0
    velocity = velocity ** 2

    #find average velocity and height
    velocityPeaks = []
    heightPeaks = []
    for item in pos:
        try:
            velocityPeaks.append(velocity[item['maxSpeedIndex']])
            heightPeaks.append(abs(distance[item['peakIndex']] - distance[item['valleyIndex']]))
        except:
            pass

    meanvelocityPeaks = np.max(velocityPeaks)
    meanHeightPeak = np.max(heightPeaks)

    # print('mean velocity', meanvelocityPeaks)
    # print('mean height', meanHeightPeak)
    corrected = []
    for item in pos:
        # print('velocity', (velocity[item['maxSpeedIndex']]), prct * meanvelocityPeaks)
        # print('height', (abs(distance[item['peakIndex']] - distance[item['valleyIndex']])), prct * meanHeightPeak)
        try:
            #check if the peak height and veolocity are smaller than 10% of the max
            if ((velocity[item['maxSpeedIndex']]) <= prct * meanvelocityPeaks) and ((abs(distance[item['peakIndex']] - distance[item['valleyIndex']])) <= prct * meanHeightPeak):
                pass
            #check if the peak is too close to the valley
            elif abs(item['peakIndex'] - item['valleyIndex']) < minDistance:
                pass
            #check if the peak occurs after the max speed point and if the max speed point if after the valley
            elif (distance[item['peakIndex']] < distance[item['maxSpeedIndex']]):
                pass 
            #check if the max speed point if after the valley
            elif (distance[item['valleyIndex']] > distance[item['maxSpeedIndex']]):
                pass
            else:
                corrected.append(item)
        except:
            pass

    return corrected

def correctBasedonHeightandVelocityNegativePeaks(pos,distance,velocity, minDistance=5,prct=0.1):
    """Correct peaks based on height and velocity
    Args:
        pos (list): list of peaks
        distance (list): distance signal
        velocity (list): velocity signal
        minDistance (int, optional): minimum distance between peaks. Defaults to 5.
        prct (float, optional): percentage of the mean height. Defaults to 0.1.
    Returns:
        list: list of corrected peaks

    This function corrects the list of negative peaks based on certain conditions:
    1. Peaks with velocity and height smaller than a percentage (prct) of the minimum velocity and height are ignored.
    2. Peaks that are too close to their corresponding valleys (less than minDistance) are ignored.
    3. Peaks that occur after the maximum speed point or where the maximum speed point occurs after the valley are ignored.
    Peaks that satisfy all the conditions are added to the corrected list and returned.
    """
    #remove positive velocity and square it
    velocity[velocity > 0] = 0
    velocity = velocity ** 2

    #find average velocity and height
    velocityPeaks = []
    heightPeaks = []
    for item in pos:
        try:
            velocityPeaks.append(velocity[item['maxSpeedIndex']])
            heightPeaks.append(abs(distance[item['peakIndex']] - distance[item['valleyIndex']]))
        except:
            pass

    meanvelocityPeaks = np.max(velocityPeaks)
    meanHeightPeak = np.max(heightPeaks)

    # print('mean velocity', meanvelocityPeaks)
    # print('mean height', meanHeightPeak)
    corrected = []
    for item in pos:
        # print('velocity', (velocity[item['maxSpeedIndex']]), prct * meanvelocityPeaks)
        # print('height', (abs(distance[item['peakIndex']] - distance[item['valleyIndex']])), prct * meanHeightPeak)
        try:
            #check if the peak height and veolocity are smaller than 10% of the min
            if ((velocity[item['maxSpeedIndex']]) <= prct * meanvelocityPeaks) and ((abs(distance[item['peakIndex']] - distance[item['valleyIndex']])) <= prct * meanHeightPeak):
                pass
            #check if the peak is too close to the valley
            elif abs(item['peakIndex'] - item['valleyIndex']) < minDistance:
                pass
            #check if the peak occurs after the max speed point and if the max speed point if after the valley
            elif (distance[item['peakIndex']] < distance[item['maxSpeedIndex']]):
                pass 
            #check if the max speed point if after the valley
            elif (distance[item['valleyIndex']] > distance[item['maxSpeedIndex']]):
                pass
            else:
                corrected.append(item)
        except:
            pass

    return corrected

def peakFinder(rawSignal, fs=30, minDistance=5, cutOffFrequency=10, prct=0.125):
    """
    Identifies positive and negative velocity peaks in a raw signal and applies corrections 
    to refine the detected peaks based on various criteria.

    Parameters:
    ----------
    rawSignal : array-like
        The input raw signal data to analyze.
    fs : int, optional
        Sampling frequency of the signal in Hz. Default is 30.
    minDistance : int, optional
        Minimum distance (in samples) between peaks to be considered valid. Default is 5.
    cutOffFrequency : float, optional
        Cut-off frequency for the low-pass Butterworth filter applied to the signal. Default is 10 Hz.
    prct : float, optional
        Percentage threshold for selecting peaks based on their height relative to the mean height. Default is 0.125.

    Returns:
    -------
    distance : array-like
        The filtered signal representing the distance.
    velocity : array-like
        The first derivative of the distance signal, representing velocity.
    peaks : list of dict
        A list of dictionaries containing information about the corrected peaks, including their indices and properties.
    indexPositiveVelocity : list of dict
        A list of dictionaries containing information about the positive velocity peaks, including:
        - 'maxSpeedIndex': Index of the maximum speed.
        - 'maxSpeed': Value of the maximum speed.
        - 'peakIndex': Index of the peak.
        - 'valleyIndex': Index of the valley.
    indexNegativeVelocity : list of dict
        A list of dictionaries containing information about the negative velocity peaks, including:
        - 'maxSpeedIndex': Index of the maximum speed.
        - 'maxSpeed': Value of the maximum speed.
        - 'peakIndex': Index of the peak.
        - 'valleyIndex': Index of the valley.

    Notes:
    -----
    1. The function applies a low-pass Butterworth filter to smooth the raw signal.
    2. Velocity is computed as the first derivative of the smoothed signal.
    3. Peaks are identified separately for positive and negative velocities using the `scipy.signal.find_peaks` function.
    4. Several correction steps are applied to refine the detected peaks:
        - Elimination of bad neighbors based on proximity.
        - Filtering based on peak height and velocity.
        - Symmetry corrections for peak and height consistency.
        - Adjustments for edge cases (heads and tails).
    5. The function is designed to handle noisy signals and ensure robust peak detection.
    Example:
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> rawSignal = np.sin(np.linspace(0, 10, 300)) + 0.1 * np.random.randn(300)
    >>> distance, velocity, peaks, posVel, negVel = peakFinder(rawSignal, fs=30, minDistance=10, cutOffFrequency=2, prct=0.1)
    >>> print(peaks)
    """

    indexPositiveVelocity = []
    indexNegativeVelocity = []


    b, a = signal.butter(2, cutOffFrequency, fs=fs, btype='low', analog=False)

    distance = signal.filtfilt(b, a, rawSignal)  # signal.savgol_filter(rawDistance[0], 5, 3, deriv=0)
    velocity = signal.savgol_filter(distance, 5, 3, deriv=1) / (1 / fs)
    ##approx mean frequency
    # acorr = np.convolve(rawSignal, rawSignal)
    acorr = np.convolve(distance, distance)
    t0 = ((1 / fs) * np.argmax(acorr))
    sep = 0.5 * (t0) if (0.5 * t0 > 1) else 1

    deriv = velocity.copy()
    deriv[deriv < 0] = 0
    deriv = deriv ** 2

    peaks, props = signal.find_peaks(deriv, distance=sep)

    heightPeaksPositive = deriv[peaks]
    selectedPeaksPositive = peaks[heightPeaksPositive > prct * np.mean(heightPeaksPositive)]

    # for each max opening vel, identify the peaks and valleys
    for idx, peak in enumerate(selectedPeaksPositive):
        idxValley = peak - 1
        if idxValley >= 0:
            while deriv[idxValley] != 0:
                if idxValley <= 0:
                    idxValley = np.nan
                    break

                idxValley -= 1

        idxPeak = peak + 1
        if idxPeak < len(deriv):
            while deriv[idxPeak] != 0:
                if idxPeak >= len(deriv) - 1:
                    idxPeak = np.nan
                    break

                idxPeak += 1

        if (not (np.isnan(idxPeak)) and not (np.isnan(idxValley))):
            positiveVelocity = {}
            positiveVelocity['maxSpeedIndex'] = peak
            positiveVelocity['maxSpeed'] = np.sqrt(deriv[peak])
            positiveVelocity['peakIndex'] = idxPeak
            positiveVelocity['valleyIndex'] = idxValley
            indexPositiveVelocity.append(positiveVelocity)

    deriv = velocity.copy()
    deriv[deriv > 0] = 0
    deriv = deriv ** 2
    peaks, props = signal.find_peaks(deriv, distance=sep)

    heightPeaksNegative = deriv[peaks]
    selectedPeaksNegative = peaks[heightPeaksNegative > prct * np.mean(heightPeaksNegative)]

    # for each max opening vel, identify the peaks and valleys
    for idx, peak in enumerate(selectedPeaksNegative):

        idxPeak = peak - 1
        if idxPeak >= 0:
            while deriv[idxPeak] != 0:
                if idxPeak <= 0:
                    idxPeak = np.nan
                    break

                idxPeak -= 1

        idxValley = peak + 1
        if idxValley < len(deriv):
            while deriv[idxValley] != 0:
                if idxValley >= len(deriv) - 1:
                    idxValley = np.nan
                    break

                idxValley += 1

        if (not (np.isnan(idxPeak)) and not (np.isnan(idxValley))):
            negativeVelocity = {}
            negativeVelocity['maxSpeedIndex'] = peak
            negativeVelocity['maxSpeed'] = np.sqrt(deriv[peak])
            negativeVelocity['peakIndex'] = idxPeak
            negativeVelocity['valleyIndex'] = idxValley
            indexNegativeVelocity.append(negativeVelocity)

            # euristics to remove bad peaks
    # # first, remove peaks that are too close to each other
    # indexPositiveVelocityCorrected = correctPeaksPositive(indexPositiveVelocity)    
    # indexNegativeVelocityCorrected = correctPeaksNegative(indexNegativeVelocity)
    # #then, remove peaks that are too small
    # indexPositiveVelocityCorrected = correctBasedonHeight(indexPositiveVelocityCorrected, distance)
    # indexNegativeVelocityCorrected = correctBasedonHeight(indexNegativeVelocityCorrected, distance)

    # remove bad peaks
    # 1- eliminate bad neighbours
    indexPositiveVelocity = eliminateBadNeighboursPositive(indexPositiveVelocity, distance, minDistance=minDistance)
    # do it a couple of times
    indexPositiveVelocity = eliminateBadNeighboursPositive(indexPositiveVelocity, distance, minDistance=minDistance)
    indexPositiveVelocity = eliminateBadNeighboursPositive(indexPositiveVelocity, distance, minDistance=minDistance)
    #2-eliminate bad peaks based on height and velocity 
    indexPositiveVelocity = correctBasedonHeightandVelocityPositivePeaks(indexPositiveVelocity, distance, velocity.copy())

    # # 2-eliminate bad peaks based on height
    # indexPositiveVelocity = correctBasedonHeight(indexPositiveVelocity, distance)
    # # 3-eliminate bad peaks based on velocity
    # indexPositiveVelocity = correctBasedonVelocityPositive(indexPositiveVelocity, velocity.copy())

    # 1- eliminate bad neighbours
    indexNegativeVelocity = eliminateBadNeighboursNegative(indexNegativeVelocity, distance, minDistance=minDistance)
    # do it a couple of times
    indexNegativeVelocity = eliminateBadNeighboursNegative(indexNegativeVelocity, distance, minDistance=minDistance)
    indexNegativeVelocity = eliminateBadNeighboursNegative(indexNegativeVelocity, distance, minDistance=minDistance)

    #2-eliminate bad peaks based on height and velocity 
    indexNegativeVelocity = correctBasedonHeightandVelocityNegativePeaks(indexNegativeVelocity, distance, velocity.copy())
    # # 2-eliminate bad peaks based on height
    # indexNegativeVelocity = correctBasedonHeight(indexNegativeVelocity, distance)
    # # 3-eliminate bad peaks based on velocity
    # indexNegativeVelocity = correctBasedonVelocityNegative(indexNegativeVelocity, velocity.copy())
    
    peaks = correctFullPeaks(distance, indexPositiveVelocity, indexNegativeVelocity)
    peaks = correctHeadsandTails(peaks)
    peaks = correctBasedonPeakSymmetry(peaks)
    peaks = correctBasedonHeightSymmetry(peaks, distance)
    peaks = correctBasedonHeightSymmetryRatio(peaks, distance)

   
    

    return distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity
