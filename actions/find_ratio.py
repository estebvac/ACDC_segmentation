import numpy as np


def find_ratio(dataset):
    """
    This ROI generator follows the structure FILE/FILE+extension
    and generates a ROI file with the name   FILE/FILE+_norm+extension
    Parameters
    ----------
    path        Images Path

    Returns     Generated ROI images
    -------

    """

    total_hist = None
    for counter in range(len(dataset)):
        hist, _ = np.histogram(dataset[counter][1].numpy(), bins=[0, 1, 2, 3, 4])

        if total_hist is None:
            total_hist = hist
        else:
            total_hist += hist

    total_hist = total_hist/np.sum(total_hist)
    return total_hist