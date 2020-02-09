from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia


class SegmentationTransforms:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Sometimes(0.3, iaa.Affine(
                rotate=(-10, 10),
                shear=(-10, 10),
                order=0,
                cval=0,
                mode='constant'
            )),
            iaa.Crop(percent=(0, 0.1)),
            iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(0, 10.0), sigma=(4.0, 6.0)))
        ])

    def __call__(self, img, label):
        original_shape = img.shape

        # Reshape to the imgaug shape:
        img = img.reshape(img.shape[1:])
        label = label.reshape(label.shape[1:])

        # Concatenate the labels so apply a single transform
        images = np.concatenate((img, label), axis=2)
        images_aug = self.aug(image=images)

        # Decode the transformed images
        n_slices = original_shape[3]
        img_aug, label_aug = images_aug[:,:,:n_slices], images_aug[:,:,n_slices:]
        img_aug = img_aug.reshape(original_shape)
        label_aug = label_aug.reshape(original_shape)
        label_aug = np.clip(np.round(label_aug), np.min(label), np.max(label))
        '''
        plt.subplot(1, 2, 1)
        plt.imshow(img_aug[0,:,:,1])
        plt.subplot(1, 2, 2)
        plt.imshow(label_aug[0, :, :, 1])
        plt.show()
        '''
        return img_aug, label_aug



