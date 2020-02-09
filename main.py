from preprocessing.preprocessing import *
from file_management.read_images import *
from file_management.split_dataset import split_dataset
from preprocessing.find_ROI import find_all_ROI
from utils.utils import MRI_DataPatchLoader
from transformations.transformation import SegmentationTransforms
from models.unet_instance_norm import UnetModel
from actions.train_model import train_model
from torch.utils.data import DataLoader
from actions.find_ratio import find_ratio


# Define path location and reference image
path_train = r'C:\Users\esteb\Documents\acdc\training'
reference_img = r'C:\Users\esteb\Documents\acdc\reference.nii.gz'

# Read the images
img_dataframe = read_image_name(path_train, '.nii.gz')

# Preprocess all the images Spacing and Histogram Matching
img_dataframe = normalize_all_images(img_dataframe, reference_img)

img_dataframe = find_all_ROI(img_dataframe, debug=False)


valid_size = 0.2
test_size = 0.2
input_dictionary = split_dataset(img_dataframe,valid_size, test_size)

# Setup the options:
options = {}
options['patch_size'] = (128, 128, 4)
options['sampling_step'] = (16, 16, 2)
options['normalize'] = True
options['batch_size'] = 64


train_transforms = SegmentationTransforms()

print('Training data: ')
training_dataset = MRI_DataPatchLoader(input_data=input_dictionary['input_train_data'],
                                       labels=input_dictionary['input_train_labels'],
                                       rois=input_dictionary['input_train_rois'],
                                       patch_size=options['patch_size'],
                                       sampling_step=options['sampling_step'],
                                       normalize=options['normalize'],
                                       sampling_type = 'balanced+roi',
                                       transform=train_transforms
                                       )

training_dataloader = DataLoader(training_dataset,
                                 batch_size=options['batch_size'],
                                 shuffle=True)


print('_________________________________________________')
print('Validation data: ')
validation_dataset = MRI_DataPatchLoader(input_data=input_dictionary['input_val_data'],
                                        labels=input_dictionary['input_val_labels'],
                                        rois=input_dictionary['input_val_rois'],
                                        patch_size=options['patch_size'],
                                        sampling_step=options['sampling_step'],
                                        normalize=options['normalize'])

validation_dataloader = DataLoader(validation_dataset,
                                   batch_size=options['batch_size'],
                                   shuffle=True)

ratio = find_ratio(training_dataset)
weights = 1 / ratio



model = UnetModel(in_chans=1, out_chans=4, chans=64, num_pool_layers=3, drop_prob=0.1)


options['gpu_use'] = False
options['num_epochs'] = 200
options['model_name'] = 'unet_tissue_segmentation'
options['save_path'] = 'models' #'/content/gdrive/My Drive/misa_model_41/'
options['patience'] = 20
options['weights'] = weights

options['lr'] = 0.0001
train_model(model, options, training_dataloader, validation_dataloader)