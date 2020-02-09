import skimage.io as io
import numpy as np
import SimpleITK  as sitk
import cv2
import matplotlib.pyplot as plt
from file_management.read_images import save_with_metadata_itk
from tqdm import tqdm



def find_ROI(img_itk, padding =64, debug=False):
    # Apply 3D Gaussian Blur to the image
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(5)
    filtered_img = gaussian.Execute(img_itk)

    # Get the average response in the axial view
    filtered_np = sitk.GetArrayFromImage(filtered_img)
    img_mean = np.mean(filtered_np, axis=0)
    # Remove the borders that contain fat that is brighter
    cropped_mean = img_mean[50:-50, 50:-50]

    # Locate the heart in the axial view
    #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(cropped_mean)
    normalizedImg = np.zeros_like(cropped_mean)
    cv2.normalize(cropped_mean, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    normalizedImg = normalizedImg.astype(np.uint8())
    circles = cv2.HoughCircles(normalizedImg, cv2.HOUGH_GRADIENT, 1, 10,
                               param1=20, param2=10, minRadius=15, maxRadius=25)
    if circles is  None:
        circles = [[[0, 0, 0]]]
    elif len(circles[0, :])>1:
        counter = 0
        circles_out = []
        features = []


        ret, thresh1 = cv2.threshold(normalizedImg, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        center = np.array(list(normalizedImg.shape))/2
        for i,circ in enumerate(circles[0, :]):
            circ = np.uint16(np.around(circ))
            intensity = 0
            try:
                intensity = normalizedImg[circ[1], circ[0]]
            except:
                pass
            if intensity > ret:
                circles_out.append(circ)
                position = np.array((circ[1], circ[0]))
                dist = np.linalg.norm(position-center)
                features.append([counter ,intensity, dist])
                counter += 1

        features_np = np.array(features)
        features_np = features_np[features_np[:,1].argsort()] [-min(3,len(features)):,:]
        features_np = features_np[features_np[:,2].argsort()]
        roi = int(features_np[0,0])
        circles = [[circles_out[roi]]]


    circles = np.uint16(np.around(circles))
    image = cv2.cvtColor(normalizedImg, cv2.COLOR_GRAY2BGR)
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        # draw the method2
        roi_center = [i[1]+50, i[0]+50]
    if debug:
        plt.imshow(image, aspect=1)
        plt.show()

    roi_img = np.zeros_like(filtered_np)
    shape_img = roi_img.shape
    roi_img[:, max(0,roi_center[0]-padding ): min(shape_img[1], roi_center[0]+padding),
               max(0,roi_center[1]-padding ): min(shape_img[2], roi_center[1]+padding)] = 1

    return image, roi_center, roi_img


def find_all_ROI(img_df, padding =64, debug=False):
    number_images = len(img_df)
    rois = []
    pbar = tqdm(range(number_images))
    for counter in pbar:
        im_path = img_df['dias'][counter]
        img_itk = sitk.ReadImage(im_path)
        roi_example, roi_center, roi_img = find_ROI(img_itk, padding =padding, debug=debug)
        save_path = im_path[:-7] + '_loc.jpg'
        cv2.imwrite(save_path, roi_example)
        filename = im_path[:-11] + 'roi.nii.gz'
        save_with_metadata_itk(roi_img, im_path, filename)
        rois.append(filename)
        pbar.set_description("Findiing ROIs ")

    pbar.close()
    img_df['ROI'] = rois
    return img_df