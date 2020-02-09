import numpy as np
import pandas as pd

def __dice(volume_counter, mask_counter):
    """
    Calcuate the dice index of one region of interest
    Parameters
    ----------
    volume_counter:     segmentation result
    mask_counter :      ground truth image

    Returns
    -------
    Dice score of the region of interest
    """
    num = 2 * (volume_counter * mask_counter).sum()
    den = volume_counter.sum() + mask_counter.sum()
    dice_tissue = num / den
    return dice_tissue

def calculate_dices(tissues: int, volume, gt) -> dict:
    """
    Calculates the dice score of all the regions segmented
    Parameters
    ----------
    tissues:        Number of regions to analize
    volume:         Segmentation result
    gt:             Ground truth image

    Returns
    -------
    The dice score of all the regions segmented.

    """
    dices_per_tissue = np.zeros([tissues, ])
    for tissue_id in range(1, tissues + 1):
        volume_counter = 1 * (volume == tissue_id)
        mask_counter = 1 * (gt == tissue_id)
        dice_tissue = __dice(volume_counter, mask_counter)
        dices_per_tissue[tissue_id - 1] = dice_tissue

    return dices_per_tissue

