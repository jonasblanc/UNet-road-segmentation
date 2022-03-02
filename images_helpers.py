from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import re
import torch
from torch import Tensor
from sklearn.metrics import f1_score
from PIL import Image

# ================ FILE & FOLDER PATHS ================ #

# Directories
TRAINING_DIR = "data/training/"
TEST_DIR = "data/test/"
SUBMISSION_DIR = "data/test_set_images/"
MODEL_DIR = "model/"

# Images
TRAINING_IMAGE_DIR = TRAINING_DIR + "images/"
TRAINING_GT_DIR = TRAINING_DIR + "groundtruth/"
AUGMENTED_IMAGE_TRAIN_DIR = TRAINING_DIR + "augmented_images/"
AUGMENTED_GT_TRAIN_DIR = TRAINING_DIR + "augmented_groundtruth/"
AUGMENTED_IMAGE_TEST_DIR = TEST_DIR + "augmented_images/"
AUGMENTED_GT_TEST_DIR = TEST_DIR + "augmented_groundtruth/"

# ================ Constants ================ #
GT_THRESHOLD = 0.25
PATCH_SIZE = 16
PIXEL_DEPTH = 255

# ================ Image loading / Writing ================ #

def write_images_to_dir(directory, imgs, names):
    '''
    Write the imgs in names files in directory
    Input:
        imgs: list of np.array
        directory: path to directory
        names: list of names
    '''
    if not os.path.isdir(directory):
        os.mkdir(directory)
    for img_array, name in zip(imgs, names):
        if(len(img_array.shape) == 3):
            img = img_float_to_uint8(img_array)
            img = Image.fromarray(img, 'RGB')
        if(len(img_array.shape) == 2):
            img = from_mask_to_img(img_array)

        img.save(directory + name)

    print(f"{len(imgs)} images saved")

def load_image(filename):
    '''Returns a numpy array of the image 
    Input:
        filename: string: name of the file
    Output:
        image: np.array(height,width,channel_count)'''
    data = mpimg.imread(filename)
    return data

def load_images(dir_name, max_image_count=math.inf):
    '''Loads all images in the provided directory:
    Input:
        dir_name: string: the name of the directory
        max_image_count: int: the maximum number of images to load
    Output:
        images: list(np.array(height,width,channel_count))
        aimges_name: list(image_name)
    '''
    # Same image names in groundtruth and images (1 to 1 correspondance)
    image_filenames = os.listdir(dir_name)
    n = min(max_image_count, len(image_filenames))  # Load maximum 20 images
    imgs = [load_image(dir_name + image_filenames[i]) for i in range(n)]
    print(f"{len(imgs)} images loaded")

    return imgs, image_filenames


def load_submission_images(dir_prefix=""):
    ''' Load test images used to produce the AIcrowd predictions
    Output:
        images: list(np.array(height,width,channel_count))
    '''

    sub_dirs = os.listdir(dir_prefix + SUBMISSION_DIR)
    sub_dirs = [f for f in sub_dirs if not f.startswith(".")]
    sub_imgs = [load_image(dir_prefix + SUBMISSION_DIR + f + "/" + f + ".png")
                 for f in sub_dirs]

    return sub_imgs, sub_dirs
   
# ================ Image type manipulation ================ #

def img_float_to_uint8(img):
    '''
    Convert float img to uint8
    '''
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

def images_to_tensor(imgs):
    '''Convert a list of numpy array images (height, width, channel_count) 
    to a Tensor of (image_count, channel_count, height, width)'''

    imgs_array = np.array(imgs)
    # Converts (image_count, height, width,channel_count) to
    #(image_count, channel_count, height, width)
    imgs_array_flip = np.einsum('abcd->adbc', imgs_array)
    return Tensor(imgs_array_flip)


def gts_to_tensor(gts):
    '''Convert a list of numpy array images (height, width, channel_count) 
    to a Tensor of (image_count, height, width)'''

    GT_PIXEL_THRESHOLD = 0.5

    gts_array = np.array(gts)
    gts_array = gts_array[:, :, :, 0]
    gts_array = np.where(gts_array > GT_PIXEL_THRESHOLD, 1.0, 0.0)

    return Tensor(gts_array)

def from_mask_to_img(mask):
    '''
    Create an 3 channel img from a 1 channel np.array
    '''
    w = mask.shape[0]
    h = mask.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = mask * PIXEL_DEPTH
    color_mask[:, :, 1] = mask * PIXEL_DEPTH
    color_mask[:, :, 2] = mask * PIXEL_DEPTH

    img = Image.fromarray(color_mask, 'RGB')
    return img

# ================ Image size manipulation ================ #

def crop_imgs(imgs, cropped_size):
    """
    Crop imgs to cropped_size
    imgs: List of np.array (size_x, size_y, channels)
    cropped_size: dest size
    return list of np.array (cropped_size, cropped_size, channels)
    """
    margin_size = int((imgs[0].shape[0] - cropped_size)/2)
    imgs = [img[margin_size:-margin_size, margin_size:-margin_size]
            for img in imgs]
    return imgs


def pad_imgs(imgs, padded_size):
    """
    Add symmetric padding to imgs to reach padded_size
    imgs: List of np.array (size_x, size_y, channels)
    padded_size: dest size
    return np.array (num imgs, padded_size, padded_size, channels)
    """
    n = int((padded_size - imgs[0].shape[0])/2)
    if len(imgs[0].shape) == 2:
        imgs_extended = np.pad(imgs, ((0, 0), (n, n), (n, n)), "symmetric")
    if len(imgs[0].shape) == 3:
        imgs_extended = np.pad(imgs, ((0, 0), (n, n), (n, n), (0, 0)), "symmetric")
    return imgs_extended

# ================ Patch processing ================ #

def patch_to_label(patch):
    ''' Take an array and output corresponding label'''
    df = np.mean(patch)
    if df > GT_THRESHOLD:
        return 1
    else:
        return 0
    
def reduce_to_patches(img):
    ''' Reduce groundtruth to patches corresponding to the output'''
    reduced_img_size = img.shape[0]//PATCH_SIZE
    reduced_img = np.zeros((reduced_img_size, reduced_img_size))
    for i in range(reduced_img_size):
        for j in range(reduced_img_size):
            i_start = i * PATCH_SIZE
            j_start = j * PATCH_SIZE
            patch = img[i_start:i_start + PATCH_SIZE,
                        j_start:j_start + PATCH_SIZE]
            reduced_img[i, j] = patch_to_label(patch)

    return reduced_img

def merge_four_patches(values, input_size, wanted_size):
    '''
    Merge 4 patches from the 4 corners into one image (with possible overlap) 
    '''
    input_weight = np.ones((input_size,input_size))
    
    weights = np.zeros((wanted_size,wanted_size))
    output = np.zeros((wanted_size,wanted_size))
    
    output[0:input_size, 0:input_size] += values[0]
    output[0:input_size, -input_size:] += values[1]
    output[-input_size:, 0:input_size] += values[2]
    output[-input_size:, -input_size:] += values[3]
    
    weights[0:input_size, 0:input_size] += input_weight
    weights[0:input_size, -input_size:] += input_weight
    weights[-input_size:, 0:input_size] += input_weight
    weights[-input_size:, -input_size:] += input_weight
    
    return output/weights

def create_four_patches(imgs, size_patch):
    '''
    Create 4 patches for each imgs (one patch per corner with possible overlap)
    '''
    patches_imgs = []
    
    for img in imgs:
        patches_imgs.append(img[0:size_patch, 0:size_patch,:]) # Top left
        patches_imgs.append(img[0:size_patch, -size_patch:,:]) # Top right
        patches_imgs.append(img[-size_patch:, 0:size_patch,:]) # Bottom left
        patches_imgs.append(img[-size_patch:, -size_patch:,:]) # Bottom right
            
    return patches_imgs

# ================ Plot images ================ #

def plot_images(img1, img2):
    '''Plot 2 images side by side
    Input:
        img: np.array(height,width,channel_count)
    '''

    fig = plt.figure(figsize=(10, 7))

    fig.add_subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis('off')
    plt.title("Image1")

    fig.add_subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis('off')
    plt.title("Image2")

# ================ Submission helpers ================ #

def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts output images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            print(fn)
            f.writelines('{}\n'.format(s)
                         for s in mask_to_submission_strings(fn))
