import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_metrics(model, test_dataset, apply_sigmoid_on_model=True):
    '''
    Returns the average accuracy and f1-score of the model on the test images
    Input:
        model: the neural net model
        test_imgs: Tensor(image_count,channel_count,image_height,image_width): the images to test the model on
        gt_imgs: Tensor(image_count,image_height,image_width): the groundtruth values for the test images
        apply_sigmoid_on_model: boolean: whether a sigmoid function should be applied on the model for the predictions

    Output:
        average_accuracy: float
        average_f1_score: float
    '''

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    accuracy_sum = 0
    f1_score_sum = 0
    n = len(test_dataset)
    with torch.no_grad():
        model.eval()
        for test_img, gt_img in test_dataset:

            test_img = test_img[None, :]  # Add batch axis
            gt_flatten = gt_img.flatten()

            test_img = test_img.to(DEVICE)
            
            y_pred = model(test_img)
            y_pred = y_pred.cpu()
            if apply_sigmoid_on_model:
                y_pred = torch.sigmoid(y_pred)

            # apply tresholding
            y_pred = np.where(y_pred > 0.5, 1, 0).flatten()

            accuracy = (np.array(y_pred) == np.array(
                gt_flatten)).sum()/len(y_pred)
            f1_score_value = f1_score(y_pred, gt_flatten)

            accuracy_sum += accuracy
            f1_score_sum += f1_score_value

    average_accuracy = accuracy_sum/n
    average_f1_score = f1_score_sum/n

    return average_accuracy, average_f1_score
