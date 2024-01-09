from sklearn.preprocessing import StandardScaler
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize
import pandas as pd
from matplotlib.image import imread
from skimage.io import imread_collection
from PIL import Image
import seaborn as sns
from sklearn import decomposition, preprocessing, svm
import sklearn.metrics as metrics #confusion_matrix, accuracy_score
from time import sleep 
from tqdm.notebook import tqdm
import os
sns.set()

#Dataset that should go with Alzheimer label
very_mild = glob(r'/Users/spangilinan/Downloads/Dataset')
mild = glob(r'/Users/spangilinan/Downloads/Dataset')
moderate = glob(r'/Users/spangilinan/Downloads/Dataset')

#Dataset without Alzheimer
non = glob(r'/Users/spangilinan/Downloads/Dataset')

#testing fuctionality 
print(non[1])
def view_image(directory):
    img = mpimg.imread(directory)
    plt.imshow(img)
    plt.title(directory)
    plt.axis('off')
    print(f'Image shape:{img.shape}')
    return img

print('One of the data in Non Alzheimer Folder')
view_image(non[1])

print('Alzheimer Patient\'s Brain')
view_image(moderate[1])