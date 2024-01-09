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
very_mild = glob(r'/Users/spangilinan/Downloads/Dataset/Very_Mild_Demented')
mild = glob(r'/Users/spangilinan/Downloads/Dataset/Mild_Demented')
moderate = glob(r'/Users/spangilinan/Downloads/Dataset/Moderate_Demented')

#Dataset without Alzheimer
non = glob(r'/Users/spangilinan/Downloads/Dataset/Non_Demented')

#testing fuctionality 
print(non[1])
def view_image(directory):
    img = mpimg.imread(directory)
    plt.imshow(img)
    plt.title(directory)
    plt.axis('off')
    print(f'Image shape:{img.shape}')
    return img

print('Data Sample of a Patient\'s Brain WITHOUT Alzheimers')
view_image(non[1])

print('Data Sample of a Patient\'s Brain WITH Alzheimers')
view_image(moderate[1])

#detect Alzheimers
def detect_alzheimers (self):
    #List where arrays shall be stored
    resized_image_array=[]
    #List that will store the answer if an image is female (0) or male (1)
    resized_image_array_label=[]

    width = 256
    height = 256
    new_size = (width,height) #the data is just black to white 

    #Iterate over pictures and resize them to 256 by 256
    def resizer(image_directory):
        for file in image_directory: #tried with os.listdir but could work with os.walk as well
            img = Image.open(file) #just putting image_directory or file does not work for google colab, interesting. 
            #preserve aspect ratio
            img = img.resize(new_size)
            array_temp = np.array(img)
            shape_new = width*height
            img_wide = array_temp.reshape(1, shape_new)
            resized_image_array.append(img_wide[0])
            if image_directory == non:
                resized_image_array_label.append(0)
            else:
                resized_image_array_label.append(1)

    ALZ = very_mild + mild + moderate
    resizer(non)
    resizer(ALZ)

    print(len(non))
    print(len(ALZ)) #data are well transformed. Let's conduct SVM
    print(len(resized_image_array))
    print(resized_image_array[1])

    #split the data to test and training
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(resized_image_array, resized_image_array_label, test_size = 0.2)

    #train SVM model
    #from sklearn import svm
    clf = svm.SVC(kernel = 'linear')
    clf.fit(train_x, train_y)
    #store predictions and ground truth
    y_pred = clf.predict(train_x)
    y_true = train_y

    #assess the performance of the SVM with linear kernel on Training data
    print('Accuracy : ', metrics.accuracy_score(y_true, y_pred))
    print('Precision : ', metrics.precision_score(y_true, y_pred))
    print('Recall : ', metrics.recall_score(y_true, y_pred))
    print('f1 : ', metrics.f1_score(y_true, y_pred)) 
    print('Confusion matrix :', metrics.confusion_matrix(y_true, y_pred)) #The training seems to be done with high accuracy on training data.

    #Now, use the SVM model to predict Test data
    y_pred = clf.predict(test_x)
    y_true = test_y

    #assess the performance of the SVM with linear kernel on Testing data
    print('Accuracy : ', metrics.accuracy_score(y_true, y_pred))
    print('Precision : ', metrics.precision_score(y_true, y_pred))
    print('Recall : ', metrics.recall_score(y_true, y_pred))
    print('f1 : ', metrics.f1_score(y_true, y_pred)) 
    print('Confusion matrix :', metrics.confusion_matrix(y_true, y_pred)) #Having high training data accuracy might mean that it is having some overfitting

#classify Alzheimers
def classify_alzheimers(self):
    #List where arrays shall be stored
    resized_image_array=[]
    #List that will store the answer if an image is female (0) or male (1)
    resized_image_array_label=[]

    width = 256
    height = 256
    new_size = (width,height) #the data is just black to white 

    #Iterate over pictures and resize them to 256 by 256
    def resizer(image_directory):
        for file in image_directory: #tried with os.listdir but could work with os.walk as well
            img = Image.open(file) #just putting image_directory or file does not work for google colab, interesting. 
            #preserve aspect ratio
            img = img.resize(new_size)
            array_temp = np.array(img)
            shape_new = width*height
            img_wide = array_temp.reshape(1, shape_new)
            resized_image_array.append(img_wide[0])
            if image_directory == non:
                resized_image_array_label.append(0)
            elif image_directory == very_mild:
                resized_image_array_label.append(1)
            elif image_directory == mild:
                resized_image_array_label.append(2)
            else:
                resized_image_array_label.append(3)

    resizer(non)
    resizer(very_mild)
    resizer(mild)
    resizer(moderate)

    #split the data to test and training
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(resized_image_array, resized_image_array_label, test_size = 0.2)

    #train SVM model
    #from sklearn import svm
    clf = svm.SVC(kernel = 'linear')
    clf.fit(train_x, train_y)
    #store predictions and ground truth
    y_pred = clf.predict(train_x)
    y_true = train_y

    #assess the performance of the SVM with linear kernel on Training data
    print('Accuracy : ', metrics.accuracy_score(y_true, y_pred))

    #Now, use the SVM model to predict Test data
    y_pred = clf.predict(test_x)
    y_true = test_y

    #assess the performance of the SVM with linear kernel on Testing data
    print('Accuracy : ', metrics.accuracy_score(y_true, y_pred))