from typing import Callable, Tuple
import numpy as np

from ..sets import IntuitionisticFuzzySet


def threshold(image: np.ndarray, measure_caller: Callable, *, is_distance: bool = True, l: float = 0.2, **kwargs) -> Tuple[np.ndarray, bool]:
    """ Thresholds the input image following the proposed method by T. Chaira and A.K. Ray, from the related article:
    "Threshold selection using fuzzy set theory".

    For each thresold value t [0, 255], the set S1 of the ideally thresholded image is calculated and 
    the set S2 of thresholded image with value t.
    The selected threshold value is the one with the min/max (Depending on is_distance) measure
    between the sets for each threshold value t.

    Parameters
    ----------
    image : np.ndarray
        Single channel input image.
    measure_caller : Callable
        The measure function to use when measuring the sets of the ideally thresholded image and the current image set.
    is_distance : bool
        If the measure provided is a distance or a similarity. Used to pick the best measure calculated.
    l : float
        Used to calculate the membership values of the thresholded image with value t.
    **kwargs : additional arguments
        Passed to the measure_caller.
        
    Returns
    -------
    thresholded_image : np.ndarray
        Thresholded image.
    threshold_value : np.float64
        Threshold value.
    """
    x, y = image.shape
    img = image.copy().flatten()
    mx = np.zeros_like(img, dtype=np.float64)
    fmax = np.max(img)
    fmin = np.min(img)
    c = 1 / (fmax - fmin)

    measures = []
    for t in range(256):
        low_count = np.sum(img <= t)
        m0 = (low_count * t) / low_count
        if np.isnan(m0) or not np.isfinite(m0):
            m0 = 0
        high_count = np.sum(img > t)
        m1 = (high_count * t) / high_count
        if np.isnan(m1) or not np.isfinite(m1):
            m1 = 0

        np.putmask(mx, img >= t, np.exp(-c * np.abs(img - m0)))
        np.putmask(mx, img < t, np.exp(-c * np.abs(img - m1)))

        A = IntuitionisticFuzzySet(
            mx, 
            (1.0 - mx) - ((1.0 - mx) / (1.0 + l * mx))
        )
        B = IntuitionisticFuzzySet([1.0] * len(A))
        measures.append(
            measure_caller(A, B, **kwargs)
        )
    
    t = np.argmin(measures) if is_distance else np.argmax(measures)
    mask = np.zeros_like(img)
    mask[img >= t] = 255
    return mask.reshape((x, y)), t

