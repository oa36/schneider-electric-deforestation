import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image

import torch

#plot an image
def plot_image(image_path):
    img = cv2.imread(image_path)
    plt.imshow(img)
    plt.show()
    
#catch corrupted images
def bad_files(paths):
    bad_images = []
    for filename in tqdm(paths):
        if filename.endswith('.png'):
            try:
                img = Image.open(filename) # open the image file
                img.verify() # verify that it is, in fact an image 
            except (IOError, SyntaxError) as e:
                bad_images.append(filename)
                print('Bad file:', filename) # print out the names of corrupt files                
    return(bad_images)

#calculate f1_score per epoch
def calc_f1_score(y_pred, y_test):
    #first convert the softmax output to probalities and take the max
    y_pred_softmax = torch.log_softmax(y_pred, 1)
    _, y_pred_tags = torch.max(y_pred_softmax, 1)

    #performance metrics
    tp = (y_test * y_pred_tags).sum().to(torch.float32)
    tn = ((1 - y_test) * (1 - y_pred_tags)).sum().to(torch.float32)
    fp = ((1 - y_test) * y_pred_tags).sum().to(torch.float32)
    fn = (y_test * (1 - y_pred_tags)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)    
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    
    return f1

def get_key(key):
    try:
        return int(key)
    except ValueError:
        return key