import SimpleITK as sitk
import os
import numpy as np


def find_ratio(path: str, extension: str = '.nii.gz'):
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
    total_hist1 = None
    for scan_id in os.listdir(path):
        im_path = os.path.join(path, scan_id, scan_id+ '_seg'  + extension)
        print('Getting image : ', im_path)
        image_itk = sitk.ReadImage(im_path)
        image_np = sitk.GetArrayFromImage(image_itk)
        hist, _ = np.histogram(image_np, bins=[0, 1, 2, 3, 4])
        hist1 = get_histogram(image_np)
        if total_hist is None:
            total_hist = hist
            total_hist1 = hist1
        else:
            total_hist += hist
            total_hist1 += hist1

    total_hist = total_hist/np.sum(total_hist)
    total_hist1 = total_hist1 / np.sum(total_hist1)
    return total_hist,  total_hist1


def get_histogram(image):
    bins = np.unique(image)
    hist = np.zeros_like(bins)
    count = 0
    for i in np.unique(image):
        hist[count] = np.sum(image==i)

    return hist
