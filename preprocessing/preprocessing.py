import SimpleITK as sitk
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

def histogram_matching(fixed, moving):
    """

    Parameters
    ----------
    fixed:      Base image to match all the MRI volumes
    moving      Image to mach

    Returns
    -------
    The histogram matched image result.
    """
    matcher = sitk.HistogramMatchingImageFilter()
    if (fixed.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8)):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    moving = matcher.Execute(moving, fixed)
    return moving


def normalize_spacing(image, interpolation ='nn'):
  resample = sitk.ResampleImageFilter()
  spacing_original = image.GetSpacing()
  new_spacing = (1.5625, 1.5625, spacing_original[2])
  resample.SetOutputSpacing(new_spacing)
  resample.SetOutputOrigin(image.GetOrigin())
  resample.SetOutputDirection(image.GetDirection())
  new_size = np.array(image.GetSpacing()) * np.array(image.GetSize()) / np.array(new_spacing)
  new_size = new_size.astype(np.uint32)
  resample.SetSize(new_size.tolist())
  if interpolation == 'nn':
      resample.SetInterpolator(sitk.sitkNearestNeighbor)
  else:
    resample.SetInterpolator(sitk.sitkBSpline)

  #resample.AddCommand(sitk.sitkProgressEvent, lambda: print("\rProgress: {0:03.1f}%...".format(100*resample.GetProgress()),end=''))
  #resample.AddCommand(sitk.sitkProgressEvent, lambda: sys.stdout.flush())
  out = resample.Execute(image)
  return out



def normalize_all_images(path_df, base_path: str,extension: str = '.nii.gz'):
    """
    This ROI generator follows the structure FILE/FILE+extension
    and generates a ROI file with the name   FILE/FILE+_norm+extension
    Parameters
    ----------
    path        Images Path

    Returns     Generated ROI images
    -------

    """
    number_images = len(path_df)
    fixed_itk = sitk.ReadImage(base_path,  sitk.sitkUInt16)
    sufixs = ['dias', 'sist', 'dias_gt', 'sist_gt']
    norm_df =  path_df
    pbar =tqdm(range(number_images))

    for scan_id in pbar:
        for sufix in sufixs:
            path_img = path_df[sufix][scan_id]
            moving_itk = sitk.ReadImage(path_img, sitk.sitkUInt16)
            if  not sufix.endswith('_gt'):
                moving_itk = normalize_spacing(moving_itk)
                moving_matched_itk = histogram_matching(fixed_itk, moving_itk)
            else:
                moving_matched_itk = normalize_spacing(moving_itk, interpolation ='nn')

            out_path = path_img[:-7] + '_norm' + extension
            sitk.WriteImage(moving_matched_itk, out_path)
            norm_df.at[scan_id, sufix] = path_df[sufix][scan_id][:-7] + '_norm' + extension

        pbar.set_description("Normalization and Spacing ")
    pbar.close()

    return norm_df
