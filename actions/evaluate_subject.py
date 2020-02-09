from utils.utils import *
import ants
from metrics.metrics import *
import pandas as pd
import os
from scipy.special import softmax
import SimpleITK as sitk


def evaluate_subject(scan_path: str, scan_name: str, options: dict, model):
    # Read the image to analize:
    scan = ants.image_read(os.path.join(scan_path, scan_name + ".nii.gz"))
    scan_np = scan.numpy()

    # define the torch.device
    device = torch.device('cuda') if options['gpu_use'] else torch.device('cpu')

    # Create the patches of the image to evaluate
    infer_patches, coordenates = get_inference_patches(scan_path=scan_path,
                                                       input_data=[scan_name + "_norm.nii.gz"],
                                                       roi=scan_name + "_ROI.nii.gz",
                                                       patch_shape=options['patch_size'],
                                                       step=options['sampling_step'],
                                                       normalize=options['normalize'])

    # Get the shape of the patches
    sh = infer_patches.shape
    segmentation_pred = np.zeros((sh[0], 4, sh[2], sh[3], sh[4]))
    batch_size = options['batch_size']

    # model Evaluation
    model.eval()
    b = 0
    with torch.no_grad():
        for b in range(0, len(segmentation_pred), batch_size):
            x = torch.tensor(infer_patches[b:b + batch_size]).to(device)
            pred = model(x)
            # save the result back from GPU to CPU --> numpy
            segmentation_pred[b:b + batch_size] = pred.cpu().numpy()

    # reconstruct image takes the inferred patches, the patches coordenates and the image size as inputs
    all_probs = np.zeros(scan_np.shape + (4,))
    for i in range(4):
        all_probs[:, :, :, i] = reconstruct_image(segmentation_pred[:, i], coordenates, scan.shape)

    segmented = np.argmax(all_probs, axis=3).astype(np.uint8)

    # Create a nifti image
    segm_img = ants.from_numpy(segmented)
    segm_img = ants.copy_image_info(scan, segm_img)

    # Save the segmentation mask
    output_name = os.path.join(scan_path, scan_name + '_result.nii.gz')
    ants.image_write(segm_img, output_name)

    return segm_img


def evaluate_set(test_samples: dict, options, model):
    """
    Evaluate for all the volumes in the test set
    Parameters
    ----------
    test_samples:   Dictionary that contains the volumes to evaluate
    options:        Options of the model
    model:          CNN model

    Returns
    -------
    dice_scores:    all the calculated DICE indexes.
    """
    all_dices = np.zeros((len(test_samples), 3))
    for index, sample in zip(range(len(test_samples)), test_samples):
        print('Evaluation volume: ', sample)
        scan_path = os.path.join(options['val_path'], sample)
        segm_img = evaluate_subject(scan_path, sample, options, model)
        gt_img = ants.image_read(os.path.join(scan_path, sample + '_seg.nii.gz'))
        all_dices[index, :] = calculate_dices(3, segm_img.numpy(), gt_img.numpy())

    dice_scores = pd.DataFrame(all_dices, columns=['CSF', 'GM', 'WM'], index=test_samples)

    return dice_scores


def evaluate_test_subject(scan_path: str, scan_name: str, options: dict, model):
    # Read the image to analize:
    scan = ants.image_read(os.path.join(scan_path, scan_name + ".nii.gz"))
    scan_np = scan.numpy()

    # define the torch.device
    device = torch.device('cuda') if options['gpu_use'] else torch.device('cpu')

    # Create the patches of the image to evaluate
    infer_patches, coordenates = get_inference_patches(scan_path=scan_path,
                                                       input_data=[scan_name + "_norm.nii.gz"],
                                                       roi=scan_name + "_ROI.nii.gz",
                                                       patch_shape=options['patch_size'],
                                                       step=options['sampling_step'],
                                                       normalize=options['normalize'])

    # Get the shape of the patches
    sh = infer_patches.shape
    segmentation_pred = np.zeros((sh[0], 4, sh[2], sh[3], sh[4]))
    batch_size = options['batch_size']

    # model Evaluation
    model.eval()
    b = 0
    with torch.no_grad():
        for b in range(0, len(segmentation_pred), batch_size):
            x = torch.tensor(infer_patches[b:b + batch_size]).to(device)
            pred = model(x)
            # save the result back from GPU to CPU --> numpy
            segmentation_pred[b:b + batch_size] = pred.cpu().numpy()

    # reconstruct image takes the inferred patches, the patches coordenates and the image size as inputs
    all_probs = np.zeros(scan_np.shape + (4,))
    for i in range(4):
        all_probs[:, :, :, i] = reconstruct_image(segmentation_pred[:, i], coordenates, scan.shape)

    segmented = np.argmax(all_probs, axis=3).astype(np.uint8)

    # Create a nifti image
    segm_img = ants.from_numpy(segmented)
    segm_img = ants.copy_image_info(scan, segm_img)

    # Save the segmentation mask
    output_name = os.path.join(scan_path, scan_name + '_result.nii.gz')
    ants.image_write(segm_img, output_name)

    return segm_img


def _avd_tissue(volume, gt):
    difference = np.abs(np.sum(volume) - np.sum(gt)).astype(np.float)
    total = np.sum(gt).astype(np.float)
    avd = difference / total
    return avd


def absolute_volumetric_difference(tissues, volume: np.array, gt: np.array):
    volume = volume.astype(np.uint8)
    gt = gt.astype(np.uint8)
    avd_per_tissue = np.zeros([tissues, ])
    for tissue_id in range(1, tissues + 1):
        volume_counter = 1 * (volume == tissue_id)
        mask_counter = 1 * (gt == tissue_id)
        avd_tissue = _avd_tissue(volume_counter, mask_counter)
        avd_per_tissue[tissue_id - 1] = avd_tissue

    return avd_per_tissue


# Hausdorf distance
def compute_hausdorf(gt_itk, pred):
    gt = sitk.GetArrayFromImage(gt_itk)
    h_distances = np.zeros((np.max(gt),))
    for i in range(3):
        hd = sitk.HausdorffDistanceImageFilter()
        gt_new = np.zeros_like(gt, dtype=np.uint8)
        segm_new = np.zeros_like(pred, dtype=np.uint8)
        gt_new[gt == i + 1] = 1
        segm_new[pred == i + 1] = 1
        gt_itk = sitk.GetImageFromArray(gt_new)
        gt_itk.CopyInformation(gt_itk)
        segm_itk = sitk.GetImageFromArray(segm_new)
        segm_itk.CopyInformation(gt_itk)
        hd.Execute(gt_itk, segm_itk)
        h_distances[i] = hd.GetHausdorffDistance()
    return h_distances


def evaluate_test_set(test_samples: dict, options, model):
    """
    Evaluate for all the volumes in the test set
    Parameters
    ----------
    test_samples:   Dictionary that contains the volumes to evaluate
    options:        Options of the model
    model:          CNN model

    Returns
    -------
    dice_scores:    all the calculated DICE indexes.
    """
    all_dices = np.zeros((len(test_samples), 3))
    all_avd = np.zeros((len(test_samples), 3))
    all_hud = np.zeros((len(test_samples), 3))

    for index, sample in zip(range(len(test_samples)), test_samples):
        print('Evaluation volume: ', sample)
        scan_path = os.path.join(options['test_path'], sample)
        segm_img = evaluate_test_subject(scan_path, sample, options, model)
        gt_img_ants = ants.image_read(os.path.join(scan_path, sample + '_seg.nii.gz'))
        gt_img_sitk = sitk.ReadImage(os.path.join(scan_path, sample + '_seg.nii.gz'))
        all_dices[index, :] = calculate_dices(3, segm_img.numpy(), gt_img_ants.numpy())
        all_avd[index, :] = absolute_volumetric_difference(3, segm_img.numpy(), gt_img_ants.numpy())
        all_hud[index, :] = compute_hausdorf(gt_img_sitk, segm_img.numpy())

    dice_scores = pd.DataFrame(all_dices, columns=['CSF', 'GM', 'WM'], index=test_samples)
    avd_scores = pd.DataFrame(all_avd, columns=['CSF', 'GM', 'WM'], index=test_samples)
    hud_scores = pd.DataFrame(all_hud, columns=['CSF', 'GM', 'WM'], index=test_samples)

    return dice_scores, avd_scores, hud_scores
