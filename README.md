# Project Road Segmentation

This repo is our participation to the [road segmentation project](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/) from [aicrowd](https://www.aicrowd.com).  

The goal of this project is to train a model that can differentiate the road pixels from the background pixels of a satellite image. To achieve this goal, we are provided a training set of a 100 Google Maps images of size 400×400 (see [data](/data/training/images)) and their associated ground truths where a 1 indicates a road and a 0 a background (see [data](/data/training/groundtruth)). To evaluate the performance of our model, we are also provided 50 similar unlabelled images of size 608x608 (see [data](/data/test_set_images/)).

<img src="https://user-images.githubusercontent.com/44334351/147136390-819a1c08-6b7e-49ff-8f76-056ab69b1461.png" alt="drawing" width="400"/>
Selection of predicted images using our adapted U-Net.

### U-Net

We trained a [classical U-Net](/Unet/Original_UNet.py) as well as an [adapted version of a U-Net](/Unet/Adapted_UNet.py). The latter takes as input the 400x400 training images and predicts an image of the same size (using 1 pixel padding during convolutions). To predict the submission images, it directly predicts the 608x608 images.

### Environnement setup

All necessary pacake are listed in [requirements.txt](/requirements.txt) and can be installed using `conda` with:

```
conda install --file requirements.txt 
```

### Run

To create the final prediction execute:
```
python run.py
```
The model is downloaded in `run_model`, the predicted images are stored in `run_pred` and the prediction file is called `submission` at the project root level.

### Contributors

- Nolan Chappuis [@Nchappui](https://github.com/Nchappui)
- Antoine Masanet [@Squalene](https://github.com/Squalene)
- Jonas Blanc [@jonasblanc](https://github.com/jonasblanc)
