import numpy as np


def _cycle(opening, peak, closing):
    return {
        "openingPeakIndex": peak,
        "closingPeakIndex": peak,
        "openingValleyIndex": opening,
        "openingMaxSpeedIndex": opening,
        "closingValleyIndex": closing,
        "closingMaxSpeedIndex": closing,
        "peakIndex": peak,
    }


def test_complete_cycle_selection_keeps_periodic_cycles():
    from cycle_selection import (
        finalize_cycle_valleys,
        select_complete_cycles,
    )

    distance = np.asarray([0.0, 5.0, 10.0, 5.0, 0.0, 4.0, 9.0, 4.0, 0.0])
    velocity = np.gradient(distance)
    peaks = select_complete_cycles(
        [_cycle(0, 2, 4), _cycle(4, 6, 8)], distance, velocity
    )
    peaks = finalize_cycle_valleys(peaks, distance)

    assert [peak["peakIndex"] for peak in peaks] == [2, 6]
    assert all(
        peak["openingValleyIndex"] < peak["peakIndex"] < peak["closingValleyIndex"]
        for peak in peaks
    )
    assert all("_protectOuterValleys" not in peak for peak in peaks)


def test_closing_refinement_skips_a_shallow_shoulder():
    from cycle_selection import finalize_cycle_valleys

    distance = np.asarray([0.0, 5.0, 10.0, 5.0, 3.0, 4.0, 1.0, 4.0, 9.0, 4.0, 0.0])
    peaks = finalize_cycle_valleys(
        [_cycle(0, 2, 4), _cycle(6, 8, 10)], distance
    )

    assert peaks[0]["closingValleyIndex"] == 6
