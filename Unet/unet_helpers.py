import time
import sys 
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import numpy as np


from images_helpers import *
from metrics import *

UNET_PREFIX = "../"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


def train_model(model, opt, num_epoch, train_loader, test_dataset, lossFunc, num_testing, num_test_per_epoch, num_save_per_epoch):
    MODEL_BATCH_NAME = UNET_PREFIX + MODEL_DIR + "unet_batch_"
    MODEL_EPOCH_NAME = UNET_PREFIX + MODEL_DIR + "unet_epoch_"
    BEST_MODEL_NAME = UNET_PREFIX + MODEL_DIR + "unet_best_model"
    
    if not os.path.exists(UNET_PREFIX + MODEL_DIR):
            os.makedirs(UNET_PREFIX + MODEL_DIR)

    torch.cuda.empty_cache()
    test_cap = int(len(train_loader) / num_test_per_epoch)
    save_cap = int(len(train_loader) / num_save_per_epoch)
    
    model.train()  # train mode
    
    for e in range(num_epoch):
        epochStart = time.time()
        
        print(f"Epoch num: {e+1}")
    
        losses = []
        test_losses = []
        best_test_f1 = 0
        
        for i, (x, y) in enumerate(train_loader):
            batchStart = time.time()
            
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            opt.zero_grad()
            
            pred = model(x)[:,0,:,:].float() # Remove chanel dimension
            loss = lossFunc(pred, y)
            loss.backward()
            losses.append(loss.item())
            
            opt.step()
            
            batchStop = time.time()
            
            print(f"Batch {i+1} loss {loss}, computed in {batchStop-batchStart}s")

            # Test the model
            if i % test_cap == 0:
                batch_test,_ = torch.utils.data.random_split(test_dataset, [num_testing, len(test_dataset) - num_testing])
                accuracy, f1_score = compute_metrics(model, batch_test)
                test_losses.append((accuracy, f1_score))
                print(f"Test set metrics on {num_testing} images: acc->{accuracy} f1->{f1_score}")
                if(f1_score > best_test_f1):
                    best_test_f1 = f1_score
                    torch.save(model.state_dict(),f"{BEST_MODEL_NAME}") 
                    print(f"Best model saved for epoch {e} and batch {i}")

                model.train()

            # Save the model
            if i % save_cap == 0:
                torch.save(model.state_dict(),MODEL_BATCH_NAME + str(i)) 
                torch.save(opt.state_dict(), MODEL_BATCH_NAME + str(i) + "_opt") 
                print(f"Model {MODEL_BATCH_NAME + str(i)} saved")
        
        torch.save(model.state_dict(), MODEL_EPOCH_NAME + str(e))
        torch.save(opt.state_dict(), MODEL_EPOCH_NAME + str(e) + "_opt") 
        print(f"Model {MODEL_EPOCH_NAME + str(e)} saved")

        epochStop = time.time()
        print(f"Epoch num: {e+1}, mean loss: {np.array(losses).mean()} in {(epochStop-epochStart)}s")
        
    return model, opt, losses, test_losses

# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
def sigmoid_f1_loss(y_pred:torch.Tensor, y_true:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    y_true = torch.flatten(y_true)
    
    sig = nn.Sigmoid()
    y_pred = sig(torch.flatten(y_pred))
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    #f1.requires_grad = is_training
    return 1 - f1