import pandas as pd
import os
import skimage.io as io
import numpy as np
import SimpleITK  as sitk
import nibabel as nib


def read_image_name(general_path, file_type='.mhd'):
    """
    Read all the images of the dataset

    Parameters
    ----------
    general_path:   String Path containing the folders image and groundtruth

    Returns
    -------
    images_dataframe    Datafame containing the String path of the image and the class

    """

    #  Read the dataset
    data_tuple = []

    for folder in sorted(os.listdir(general_path)):
        if os.path.isdir(general_path + '/' + folder):
            file_found = 0
            for file in sorted(os.listdir(general_path + '/' + folder)):
                if file.endswith('_gt' + file_type):
                    folder_path = general_path + '/' + folder + '/'
                    if file_found == 0:
                        dias_gt =  folder_path + file
                        dias = dias_gt[:-10]  + file_type
                        file_found += 1
                    else:
                        sist_gt = folder_path + file
                        sist = sist_gt[:-10] + file_type

        data_tuple.append((folder, dias, sist, dias_gt, sist_gt))

    images_df = pd.DataFrame(data_tuple, columns=['Folder','dias', 'sist', 'dias_gt', 'sist_gt'])
    return images_df


def read_file_name(general_path, file_type='.mhd'):
    """
    Read all the images of the dataset

    Parameters
    ----------
    general_path:   String Path containing the folders image and groundtruth

    Returns
    -------
    images_dataframe    Datafame containing the String path of the image and the class

    """

    #  Read the dataset
    data_tuple = []

    for file in sorted(os.listdir(general_path)):
        if file.endswith(file_type):
            full_path = general_path + '/' + file
            data_tuple.append(full_path)

    images_df = pd.DataFrame(data_tuple, columns=['File'])
    return images_df


def read_images(images_dataframe):
    """
    Read all the images in a dataframe
    Parameters
    ----------
    images_dataframe:       Dataframe that contains the path of all the images to read

    Returns                 list of numpy array containing all the images
    -------

    """
    images = []
    for counter in range(len(images_dataframe)):
        img = io.imread(images_dataframe['File'][counter], plugin='simpleitk')
        images.append(img)
    return images


def save_images(file_name, atlas, base_image_path):
    """
    Save the different probability layers of atlas

    Parameters
    ----------
    file_name           name to the file to save
    atlas               numpy array multilayer to save
    base_image_path     path of the file to extract metadata

    Returns
    -------

    """
    for tissue in range(1, len(atlas)):
        img = atlas[tissue]
        save_with_metadata_itk(img, base_image_path, file_name + str(tissue) + '.nii.gz')


def save_itk(image, filename):
    """
    Simple save NIFTI function

    Parameters
    ----------
    image           array with information to save
    filename        name of the output file

    Returns
    -------

    """
    im = sitk.GetImageFromArray(image, isVector=False)
    sitk.WriteImage(im, filename, True)


def save_with_metadata_itk(image, meta_file, filename):
    """
    Save with the metadata of another file, for generated images

    Parameters
    ----------
    image               Array to save
    meta_file           File that contains the metadata
    filename            name of the file to save

    Returns
    -------

    """
    base = sitk.ReadImage(meta_file)
    im = sitk.GetImageFromArray(image, isVector=False)
    im.CopyInformation(base)
    sitk.WriteImage(im, filename, True)


def read_file_name_only(general_path, file_type='.mhd'):
    """
    Read all the images of the dataset

    Parameters
    ----------
    general_path:   String Path containing the folders image and groundtruth

    Returns
    -------
    images_dataframe    Datafame containing the String path of the image and the class

    """

    #  Read the dataset
    data_tuple = []

    for file in sorted(os.listdir(general_path)):
        if file.endswith(file_type):
            data_tuple.append(file[0:-len(file_type)])

    return np.array(data_tuple)



def save_with_metadata_4d(im, meta_file, filename):
    """
    Save with the metadata of another file, for generated images

    Parameters
    ----------
    image               Array to save
    meta_file           File that contains the metadata
    filename            name of the file to save

    Returns
    -------

    """
    base = sitk.ReadImage(meta_file)
    im.SetOrigin(base.GetOrigin())
    im.SetSpacing(base.GetSpacing())
    sitk.WriteImage(im, filename, True)
