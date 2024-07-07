from scipy.signal import butter, filtfilt, savgol_filter
from misc.nonuniform_savgol_filter import *


def filter_data(x: np.array, errorbars: bool = True) -> np.array:
    """
    Filter the data.

    Parameters
    ----------
    x : np.array
        Input data.
    errorbars : bool
        Process errorbars.

    Returns
    -------
    np.array
        Filtered data.
    """

    if errorbars:
        x = np.array([item[0] for item in x])
    else:
        if type(x) is not np.array:
            x = np.array(x)

    # Savitzky-Golay filter.
    window_length = min(30, len(x))
    polyorder = min(4, window_length - 1)

    y = savgol_filter(x, window_length, polyorder)

    # window_length = 0.5
    # polyorder = 4
    # y = nonuniform_savgol_filter(np.sort(-np.array(results["metrics"]["test_loss"])), x, window_length, polyorder)

    # scipy.signal.filtfilt.
    b, a = butter(8, 0.125)
    padlen = min(5, len(x) - 1)

    y = filtfilt(b, a, y, padlen=padlen)

    return y