import urllib.request

import os
import sys
sys.path.insert(0, 'unet/')

from Adapted_UNet import *
from images_helpers import *
from unet_helpers import *
import torch

PREDICTION_DIR = "run_pred/"
MODEL_DIR = "run_model/"
MODEL_NAME = "road_segmentation.model"
MODEL_PATH = MODEL_DIR + MODEL_NAME
MODEL_URL = "https://drive.switch.ch/index.php/s/faHDF66BgleSBKK/download"

PROBA_DROPOUT = 0
PROBA_DROPOUT_MIDDLE = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def download_model():
    '''
    If the model is not already downloaded, download it from a drive
    '''
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        print(f"Download model: {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded")

def load_model():
    '''
    Load the unet
    '''
    model = UNet(PROBA_DROPOUT, PROBA_DROPOUT_MIDDLE).to(DEVICE) 
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    return model

def sigmoid(t):
    return np.exp(-np.logaddexp(0, -t))

def create_prediction(model, sub_imgs, sub_dirs):
    '''
    Predict submission image and create submission file: submissionUnet
    '''
    sub_imgs = np.array([np.moveaxis(img, -1, 0) for img in sub_imgs])
    
    with torch.no_grad():
        model.eval()
        idx = 0

        
        files_pred = []
        for i in range(len(sub_imgs)):
            if i % 2 ==0:
                xs = sub_imgs[i:i+2]
                x = torch.tensor(xs).to(DEVICE)
                y_pred = model(x)
                y_pred=y_pred.cpu()
                for j in range(2):
                    y_pred_ = sigmoid(y_pred[j,0].detach().numpy()) 
                    y_pred_ = np.where(y_pred_ > 0.5, 1, 0)            

                    img_pred = from_mask_to_img(y_pred_)
                    pred_name = PREDICTION_DIR + sub_dirs[idx]+".png"
                    img_pred.save(pred_name)
                    idx += 1
                    files_pred.append(pred_name)
                print(f"Predicted {i} images")


        masks_to_submission("submission", *files_pred)
        print(f"Prediction file created with success: submission")

if __name__ == '__main__':
    download_model()

    print("Load model")
    model = load_model()

    print("Load submission images")
    sub_imgs, sub_dirs = load_submission_images()

    if not os.path.exists(PREDICTION_DIR):
                os.makedirs(PREDICTION_DIR)

    print("Predict images")
    create_prediction(model, sub_imgs, sub_dirs)